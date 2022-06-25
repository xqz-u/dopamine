import functools as ft
from typing import Dict, Tuple

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees
from thesis.agent import utils as agent_utils


@ft.partial(static_argnums=(0,))
def train_dqn(
    gamma: float, ts: custom_pytrees.ValueBasedTS, replay_batch: Dict[str, np.ndarray]
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def loss_fn(params: FrozenDict, ys: jnp.ndarray) -> jnp.ndarray:
        qs = jax.vmap(lambda x: ts.apply_fn(params, x))(replay_batch["state"]).squeeze()
        played_qs = jax.vmap(lambda q, a: q[a])(qs, replay_batch["action"])
        return jnp.mean(jax.vmap(ts.loss_metric)(ys, played_qs))

    # Q'_t+1 = \argmax_a Q(S_t+1, a; theta^-)
    #          (or) 0 if S_t is a terminal state,
    max_qs_tp1 = (
        jax.vmap(lambda x: ts.apply_fn(ts.target_params, x))(replay_batch["next_state"])
        .squeeze()
        .max(1)
    )
    # TD-Target (y) = R_t + \gamma^N * Q'_t+1
    td_targets = agent_utils.TD_target(
        gamma, max_qs_tp1, replay_batch["reward"], replay_batch["terminal"]
    )
    # gradient descent on L = (Q(S_t, a_t; theta) - y)^2 (or another
    # loss fn)
    loss, grads = jax.value_and_grad(loss_fn)(ts.params, td_targets)
    return loss, ts.apply_gradients(grads=grads)


@ft.partial(static_argnums=(0,))
def train_dqv(
    gamma: float,
    ts_V: custom_pytrees.ValueBasedTS,
    ts_Q: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS], ...]:
    def q_loss_fn(params: FrozenDict, ys: jnp.ndarray) -> jnp.ndarray:
        qs = jax.vmap(lambda x: ts_Q.apply_fn(params, x))(
            replay_batch["state"]
        ).squeeze()
        played_qs = jax.vmap(lambda q, a: q[a])(qs, replay_batch["action"])
        return jnp.mean(jax.vmap(ts_Q.loss_metric)(ys, played_qs))

    def v_loss_fn(params: FrozenDict, ys: jnp.ndarray) -> jnp.ndarray:
        vs = jax.vmap(lambda x: ts_V.apply_fn(params, x))(
            replay_batch["state"]
        ).squeeze()
        return jnp.mean(jax.vmap(ts_Q.loss_metric)(ys, vs))

    # V'_t+1 = V(S_t+1, a; theta^-)
    #          (or) 0 if S_t is a terminal state,
    vs_tp1 = jax.vmap(lambda x: ts_V.apply_fn(ts_V.target_params, x))(
        replay_batch["next_state"]
    ).squeeze()
    #   V TD-Target (y)
    # = Q TD-Target
    # = R_t + \gamma^N * V'_t+1
    td_targets = agent_utils.TD_target(
        gamma, vs_tp1, replay_batch["reward"], replay_batch["terminal"]
    )
    # gradient descents:
    # L_V = (V(S_t, a_t; theta) - y)^2
    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(ts_V.params, td_targets)
    # L_Q = (Q(S_t, a_t; theta) - y)^2
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(ts_Q.params, td_targets)
    return (v_loss, ts_V.apply_gradients(grads=v_grads)), (
        q_loss,
        ts_Q.apply_gradients(grads=q_grads),
    )
