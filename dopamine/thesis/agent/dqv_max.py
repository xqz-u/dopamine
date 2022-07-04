import functools as ft
import logging
from typing import Dict, Tuple

import gin
import jax
import numpy as np
from attrs import define, field
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, types
from thesis.agent import dqn, dqv
from thesis.agent import utils as agent_utils

logger = logging.getLogger(__name__)


@ft.partial(jax.jit, static_argnums=(0,))
def train_dqv_max(
    gamma: float,
    q_ts: custom_pytrees.ValueBasedTS,
    v_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def q_loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = q_ts.apply_fn(params, replay_batch["state"])
        chosen_qs = jax.vmap(lambda head_qs, a: head_qs[a])(qs, replay_batch["action"])
        return q_ts.loss_metric(v_targets, chosen_qs).mean()

    def v_loss_fn(params: FrozenDict) -> jnp.ndarray:
        return v_ts.loss_metric(
            q_targets, v_ts.apply_fn(params, replay_batch["state"])
        ).mean()

    vs_st1 = v_ts.apply_fn(v_ts.params, replay_batch["next_state"])
    v_targets = agent_utils.bellman_target(
        gamma, vs_st1, replay_batch["reward"], replay_batch["terminal"]
    )

    max_qs_st1 = q_ts.apply_fn(q_ts.target_params, replay_batch["next_state"]).max(1)
    q_targets = agent_utils.bellman_target(
        gamma, max_qs_st1, replay_batch["reward"], replay_batch["terminal"]
    )

    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_ts.params)
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_ts.params)
    return (
        v_loss,
        v_ts.apply_gradients(grads=v_grads),
        q_loss,
        q_ts.apply_gradients(grads=q_grads),
    )


@gin.configurable
@define
class DQVMax(dqn.DQN):
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqn.DQN._make_train_state(self, self.V_model_def, False)
        self.train_fn = train_dqv_max

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        return dqv.DQV.train(self, experience_batch)

    @property
    def reportable(self):
        return super().reportable + ("V_model_def",)


@define
class MultiHeadEnsembleDQVMax(DQVMax):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["Q"] = dqn.MultiHeadEnsembleDQN.reassemble_Q(self)
        self.models["V"] = dqv.MultiHeadEnsembleDQV.reassemble_V(self)

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        return dqn.MultiHeadEnsembleDQN.train(self, experience_batch)
