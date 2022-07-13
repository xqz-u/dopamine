import functools as ft
from typing import Dict, Tuple

import jax
import numpy as np
from attrs import define
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, types
from thesis.agent import dqn, dqv
from thesis.agent import utils as agent_utils
from thesis.agent.dqv_max import DQVMax


# NOTE quick fixes of copy-paste abstraction to train ablation
# versions of dqvmax, being able to modify `replay_batch` for each
# trainining routine individually and to control the dimensionality
# td targets
@ft.partial(jax.jit, static_argnums=(0,))
def train_q(
    gamma: float,
    q_ts: custom_pytrees.ValueBasedTS,
    v_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def q_loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = q_ts.apply_fn(params, replay_batch["state"])
        chosen_qs = jax.vmap(lambda head_qs, a: head_qs[a])(qs, replay_batch["action"])
        return q_ts.loss_metric(v_targets, chosen_qs).mean()

    vs_st1 = v_ts.apply_fn(v_ts.params, replay_batch["next_state"])
    vs_st1 = vs_st1.squeeze()
    v_targets = agent_utils.bellman_target(
        gamma, vs_st1, replay_batch["reward"], replay_batch["terminal"]
    )
    v_targets = jnp.expand_dims(v_targets, 1)
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_ts.params)
    return q_loss, q_ts.apply_gradients(grads=q_grads)


@ft.partial(jax.jit, static_argnums=(0,))
def train_v(
    gamma: float,
    q_ts: custom_pytrees.ValueBasedTS,
    v_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def v_loss_fn(params: FrozenDict) -> jnp.ndarray:
        return v_ts.loss_metric(
            q_targets, v_ts.apply_fn(params, replay_batch["state"])
        ).mean()

    max_qs_st1 = q_ts.apply_fn(q_ts.target_params, replay_batch["next_state"]).max(1)
    q_targets = agent_utils.bellman_target(
        gamma, max_qs_st1, replay_batch["reward"], replay_batch["terminal"]
    )
    q_targets = jnp.expand_dims(q_targets, 1)
    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_ts.params)
    return v_loss, v_ts.apply_gradients(grads=v_grads)


# only Q function ensembled
@define
class MultiHeadEnsembleDQVMaxOnQ(DQVMax):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["Q"] = dqn.MultiHeadEnsembleDQN.reassemble_Q(self)

    # NOTE not calling dqn.MultiHeadEnsembleDQN.train since I want to
    # control which of rewards and terminals is expanded: can only
    # expand the one that we ensemble on
    # NOTE delayed optimized model assignment to use same version of
    # networks during tranining
    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        q_loss, new_Q = train_q(self.gamma, *self.models.values(), experience_batch)
        for k in ["reward", "terminal"]:
            experience_batch[k] = jnp.expand_dims(experience_batch[k], 1)
        v_loss, self.models["V"] = train_v(
            self.gamma, *self.models.values(), experience_batch
        )
        self.models["Q"] = new_Q
        return {"loss": {"V": v_loss, "Q": q_loss}}


# only V function ensembled
@define
class MultiHeadEnsembleDQVMaxOnV(DQVMax):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqv.MultiHeadEnsembleDQV.reassemble_V(self)

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        v_loss, new_V = train_v(self.gamma, *self.models.values(), experience_batch)
        for k in ["reward", "terminal"]:
            experience_batch[k] = jnp.expand_dims(experience_batch[k], 1)
        q_loss, self.models["Q"] = train_q(
            self.gamma, *self.models.values(), experience_batch
        )
        self.models["V"] = new_V
        return {"loss": {"V": v_loss, "Q": q_loss}}
