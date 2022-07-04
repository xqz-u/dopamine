import functools as ft
import logging
from typing import Dict, Tuple

import jax
import numpy as np
import optax
from dopamine.jax import losses
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, networks, utils
from thesis.agent import utils as agent_utils

utils.setup_root_logging(logging.DEBUG)


@ft.partial(jax.jit, static_argnums=(0,))
def train_DQVMax_multihead(
    gamma: float,
    q_ts: custom_pytrees.ValueBasedTS,
    v_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[Tuple[jnp.ndarray], Tuple[custom_pytrees.ValueBasedTS]]:
    def train_q(params: FrozenDict) -> jnp.ndarray:
        qs = q_ts.apply_fn(params, replay_batch["state"])
        chosen_qs = jax.vmap(lambda head_qs, a: head_qs[a])(qs, replay_batch["action"])
        return q_ts.loss_metric(v_targets, chosen_qs).mean()

    def train_v(params: FrozenDict) -> jnp.ndarray:
        return v_ts.loss_metric(
            q_targets, v_ts.apply_fn(params, replay_batch["state"])
        ).mean()

    expanded_rewards = jnp.expand_dims(replay_batch["reward"], 1)
    expanded_terminals = jnp.expand_dims(replay_batch["terminal"], 1)

    vs_st1 = v_ts.apply_fn(v_ts.params, replay_batch["next_state"])
    v_targets = agent_utils.bellman_target(
        gamma, vs_st1, expanded_rewards, expanded_terminals
    )

    max_qs_st1 = q_ts.apply_fn(q_ts.target_params, replay_batch["next_state"]).max(1)
    q_targets = agent_utils.bellman_target(
        gamma, max_qs_st1, expanded_rewards, expanded_terminals
    )

    v_loss, v_grads = jax.value_and_grad(train_v)(v_ts.params)
    q_loss, q_grads = jax.value_and_grad(train_q)(q_ts.params)
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

q_net = networks.MLP(features=n_actions * n_heads, hiddens=())
q_params = q_net.init(next(rng), jnp.ones(obs_shape))

v_ts = custom_pytrees.ValueBasedTS.create(
    apply_fn=lambda params, xs: jax.vmap(lambda x: v_net.apply(params, x))(xs),
    params=v_params,
    target_params=None,
    tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
    loss_metric=losses.mse_loss,
)

q_ts = custom_pytrees.ValueBasedTS.create(
    apply_fn=lambda params, xs: jax.vmap(lambda x: q_net.apply(params, x))(xs).reshape(
        (-1, n_actions, n_heads)
    ),
    params=q_params,
    target_params=q_params,
    tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
    loss_metric=losses.mse_loss,
)


for i in range(5):
    els = make_batch()
    full_train = train_DQVMax_multihead(gamma, q_ts, v_ts, els)
    if not i:
        print("COMPILED FIRST TIME")


batch = make_batch()


jax.make_jaxpr(train_DQVMax_multihead, static_argnums=(0,))(gamma, q_ts, v_ts, els)
jax.make_jaxpr(train_DQVMax_multihead_better, static_argnums=(0,))(
    gamma, q_ts, v_ts, els
)

# # compute the objective function for the Q network
# # td-targets
# vs_st1 = jax.vmap(lambda x: v_net.apply(v_params, x))(batch["next_state"])
# target_vs = jax.lax.stop_gradient(
#     bellman_target(
#         gamma,
#         vs_st1,
#         jnp.expand_dims(batch["reward"], 1),
#         jnp.expand_dims(batch["terminal"], 1),
#     )
# )
# all_qs = jax.vmap(lambda x: q_net.apply(q_params, x))(batch["state"])
# heads_qs = all_qs.reshape((-1, n_actions, n_heads))
# chosen_qs = jax.vmap(lambda head_qs, a: head_qs[a])(heads_qs, batch["action"])
# heads_losses = target_vs - chosen_qs
# heads_losses **= 2
# loss_q = heads_losses.mean()

# # compute the objective function for the V network
# qs_st1 = q_ts.apply_fn(q_ts.target_params, batch["next_state"])
# max_qst1 = qs_st1.max(1)
# target_qs = jax.lax.stop_gradient(
#     bellman_target(
#         gamma,
#         max_qst1,
#         jnp.expand_dims(batch["reward"], 1),
#         jnp.expand_dims(batch["terminal"], 1),
#     )
# )
# v_heads_losses = target_qs - v_ts.apply_fn(v_ts.params, batch["state"])
# v_heads_losses **= 2
# loss_v = v_heads_losses.mean()
