import functools as ft
import logging
from typing import Callable, Dict

import jax
import numpy as np
import optax
from dopamine.jax import losses
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, networks, utils

utils.setup_root_logging(logging.DEBUG)

rng = custom_pytrees.PRNGKeyWrap()
obs_shape = (4, 1)
n_actions, n_heads = 2, 4
test_in = jrand.uniform(next(rng), obs_shape)

net = networks.MLP(features=n_actions * n_heads, hiddens=(4,))
params = net.init(next(rng), jnp.ones(obs_shape))

ens_q = net.apply(params, test_in)
ens_q = ens_q.reshape((n_heads, n_actions))

replay_action = jrand.randint(next(rng), (), 0, 2)
terminal = jrand.randint(next(rng), (), 0, 2)
reward = jrand.uniform(next(rng), ())
s_t = jrand.uniform(next(rng), obs_shape)
s_t1 = jrand.uniform(next(rng), obs_shape)


def td_target(params, st1, reward, terminal):
    heads_qt1 = net.apply(params, st1)
    max_qt1 = heads_qt1.max(0)
    return jax.lax.stop_gradient(reward + 0.99 * (1.0 - terminal) * max_qt1)


def loss_fn(params, state, target):
    heads_qs = net.apply(params, state).reshape(n_heads, n_actions)
    ens_qs = heads_qs.mean(0)
    chosen_qs = ens_qs[replay_action]
    return jnp.power(target - chosen_qs, 2)


grad_fn = jax.value_and_grad(loss_fn)

bellman_target = td_target(params, s_t1, reward, terminal)
loss, grads = grad_fn(params, s_t, bellman_target)


batch_in = jrand.uniform(next(rng), (5,) + obs_shape)
ens_qs = jax.vmap(lambda x: net.apply(params, x))(batch_in)
ens_qs = ens_qs.reshape((-1, n_heads, n_actions))
ens_qs.max(1)
ens_qs.mean(1).max(1)

from thesis import configs
from thesis.agent import utils as agent_utils

ens_mlp_def = configs.adam_mse_ensemble_mlp(3, 2, "CartPole-v1", hiddens=())
ens_mlp = ens_mlp_def.net
ens_params = ens_mlp.init(next(rng), jnp.ones(obs_shape))

# single sample example
ens_td = ens_mlp.apply(ens_params, s_t1)
# max across action dimensions for each head
q_t1_ens = ens_td.max(-1)
# assume state is non-terminal
bellman_ens = reward + (0.99 * q_t1_ens)


ens_q = ens_mlp.apply(ens_params, s_t)
# for each head, pick q-value of action played in state s_t
ens_played_q = ens_q[:, replay_action]

# loss is now a vector, where each entry is the loss for a head on the
# sample; this cannot be differentiated by jax.value and grad.
# what the papers do is, take the mean - now jax can differentiate loss
# function
heads_losses = jnp.power(bellman_ens - ens_played_q, 2)
mean_loss = heads_losses.mean()


def q_bellman_target(
    model_call: Callable[[jnp.ndarray], jnp.ndarray],
    params: FrozenDict,
    next_state: jnp.ndarray,
    reward: jnp.ndarray,
    terminal: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    return jax.lax.stop_gradient(
        reward + gamma * model_call(params, next_state).max(1) * (1.0 - terminal)
    )


# NOTE all operations batched
@ft.partial(jax.jit, static_argnums=(0,))
def multihead_train_q(
    gamma: float, ts: custom_pytrees.ValueBasedTS, replay_batch: Dict[str, np.ndarray]
):
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = ts.apply_fn(ts.params, replay_batch["state"])
        played_qs = jax.vmap(lambda heads_qs, i: heads_qs[i])(qs, replay_els["action"])
        heads_losses = ts.loss_metric(td_targets, played_qs)
        return jnp.mean(heads_losses)

    td_targets = q_bellman_target(
        ts.apply_fn,
        ts.target_params,
        replay_batch["next_state"],
        jnp.expand_dims(replay_batch["reward"], 1),
        jnp.expand_dims(replay_batch["terminal"], 1),
        gamma,
    )
    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
    return loss, ts.apply_gradients(grads=grads)


multihead_mlp = networks.MLP(features=n_actions * n_heads, hiddens=())
multihead_mlp_params = multihead_mlp.init(next(rng), jnp.ones(obs_shape))
# NOTE batched ts
multihead_ts = custom_pytrees.ValueBasedTS.create(
    apply_fn=lambda params, xs: jax.vmap(
        lambda x: multihead_mlp.apply(multihead_mlp_params, x)
    )(xs).reshape((-1, n_actions, n_heads)),
    s_tp1_fn=None,
    params=multihead_mlp_params,
    target_params=multihead_mlp_params,
    loss_metric=losses.mse_loss,
    tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
)

# let's try with batched data now
terminals = jrand.randint(next(rng), (5,), 0, 2)
actions = jrand.randint(next(rng), (5,), 0, 2)
rewards = jrand.uniform(next(rng), (5,))
next_states = jrand.uniform(next(rng), (5,) + obs_shape)

replay_els = {
    "state": batch_in,
    "next_state": next_states,
    "action": actions,
    "terminal": terminals,
    "reward": rewards,
}

loss, ts = multihead_train_q(0.99, multihead_ts, replay_els)


# import tensorflow.compat.v1 as tf
# from batch_rl.batch_rl.multi_head import atari_helpers

# rand_test = jrand.randint(next(rng), (5, 84, 84, 4), 0, 256)

# model = atari_helpers.MultiHeadQNetwork(2, n_heads, transform_strategy="IDENTITY")
# out = model(rand_test)


# actions = replay_els["action"]
# indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
# replay_chosen_q = tf.gather_nd(out.q_heads, indices=indices)

# replay_next_qt_max = tf.reduce_max(out.q_heads, axis=1)

# is_non_terminal = 1.0 - tf.cast(replay_els["terminal"], tf.float32)
# is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
# rewards = tf.expand_dims(replay_els["reward"], axis=-1)

# target = tf.stop_gradient(rewards + (0.99 * replay_next_qt_max * is_non_terminal))

# loss = tf.losses.huber_loss(target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

# q_head_losses = tf.reduce_mean(loss, axis=0)
