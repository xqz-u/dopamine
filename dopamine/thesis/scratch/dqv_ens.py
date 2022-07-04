import functools as ft
from typing import Callable, Dict, Tuple

import jax
import numpy as np
import optax
from dopamine.jax import losses
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, networks


def bellman_target(
    gamma: float, td_targets: jnp.ndarray, reward: jnp.ndarray, terminal: jnp.ndarray
) -> jnp.ndarray:
    return reward + gamma * td_targets * (1.0 - terminal)


def q_bellman_target(
    model_call: Callable[[jnp.ndarray], jnp.ndarray],
    params: FrozenDict,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    terminal: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    return bellman_target(
        gamma, model_call(params, next_state).max(1), reward, terminal
    )


@ft.partial(jax.jit, static_argnums=(0,))
def train_DQV_multihead_singleq(
    gamma: float,
    q_ts: custom_pytrees.ValueBasedTS,
    v_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[Tuple[jnp.ndarray], Tuple[custom_pytrees.ValueBasedTS]]:
    def v_loss_fn(params: FrozenDict) -> jnp.ndarray:
        vs = v_ts.apply_fn(params, replay_batch["state"])
        # loss on each dimension - only 1 - for each head
        v_heads_loss = v_ts.loss_metric(td_targets, vs)
        # mean loss on sample among heads, then mean across samples
        return v_heads_loss.mean()

    def q_loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = jax.vmap(lambda s: q_ts.apply_fn(params, s))(replay_batch["state"])
        played_qs = jax.vmap(lambda heads_qs, i: heads_qs[i])(
            qs, replay_batch["action"]
        )
        return jax.vmap(q_ts.loss_metric)(td_targets, played_qs).mean()

    td_targets = jax.lax.stop_gradient(
        bellman_target(
            gamma,
            v_ts.apply_fn(v_ts.target_params, replay_batch["next_state"]),
            jnp.expand_dims(replay_batch["reward"], 1),
            jnp.expand_dims(replay_batch["terminal"], 1),
        )
    )
    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_ts.params)
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_ts.params)
    return (v_loss, q_loss), (
        v_ts.apply_gradients(grads=v_grads),
        q_ts.apply_gradients(grads=q_grads),
    )


make_batch = lambda: dict(
    zip(
        ["state", "next_state", "action", "terminal", "reward"],
        [
            jrand.uniform(next(rng), (5,) + obs_shape),
            jrand.uniform(next(rng), (5,) + obs_shape),
            jrand.randint(next(rng), (5,), 0, 2),
            jrand.randint(next(rng), (5,), 0, 2),
            jrand.uniform(next(rng), (5,)),
        ],
    )
)


rng = custom_pytrees.PRNGKeyWrap()
obs_shape = (4, 1)
n_actions, n_heads = 2, 4
gamma = 0.99


v_net = networks.MLP(features=n_heads, hiddens=())
v_params = v_net.init(next(rng), jnp.ones(obs_shape))

q_net = networks.MLP(features=n_actions, hiddens=())
q_params = q_net.init(next(rng), jnp.ones(obs_shape))

v_ts = custom_pytrees.ValueBasedTS.create(
    apply_fn=lambda params, xs: jax.vmap(lambda x: v_net.apply(params, x))(xs),
    params=v_params,
    target_params=v_params,
    tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
    loss_metric=losses.mse_loss,
)

q_ts = custom_pytrees.ValueBasedTS.create(
    apply_fn=q_net.apply,
    params=q_params,
    target_params=None,
    tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
    loss_metric=losses.mse_loss,
)

batch = make_batch()

full_train = train_DQV_multihead_singleq(gamma, q_ts, v_ts, batch)

td_targets = jax.lax.stop_gradient(
    bellman_target(
        gamma,
        v_ts.apply_fn(v_ts.target_params, batch["next_state"]),
        jnp.expand_dims(batch["reward"], 1),
        jnp.expand_dims(batch["terminal"], 1),
    )
)


vs = v_ts.apply_fn(v_ts.params, batch["state"])
v_heads_loss = v_ts.loss_metric(td_targets, vs)
total_loss = v_heads_loss.mean()


qs = jax.vmap(lambda s: q_ts.apply_fn(q_ts.params, s))(batch["state"])
played_qs = jax.vmap(lambda heads_qs, i: heads_qs[i])(qs, batch["action"])
jax.vmap(q_ts.loss_metric)(td_targets, played_qs).mean()

state = batch["state"][0]
next_state = batch["next_state"][0]
action = batch["action"][0]
reward = batch["reward"][0]
terminal = batch["terminal"][0]

# ensemble V target
v_st1 = v_net.apply(v_params, next_state)
v_td_target = reward + gamma * v_st1 * (1.0 - 0)

q_st = q_net.apply(q_params, state)
chosen_q = q_st[action]

q_heads_loss = losses.huber_loss(v_td_target, chosen_q)
q_loss = q_heads_loss.mean()

v_st = v_net.apply(v_params, state)
v_heads_loss = losses.huber_loss(v_td_target, v_st)
v_loss = v_heads_loss.mean()
