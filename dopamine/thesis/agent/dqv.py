import functools as ft
import logging
from typing import Dict, Tuple

import gin
import jax
import numpy as np
from attrs import define, field
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import types
from thesis.agent import base, dqn
from thesis.agent import utils as agent_utils
from thesis.custom_pytrees import ValueBasedTS

logger = logging.getLogger(__name__)


@ft.partial(jax.jit, static_argnums=(0,))
def train_dqv(
    gamma: float,
    v_ts: ValueBasedTS,
    q_ts: ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[jnp.ndarray, ValueBasedTS]:
    def v_loss_fn(params: FrozenDict) -> jnp.ndarray:
        return v_ts.loss_metric(
            td_targets, v_ts.apply_fn(params, replay_batch["state"])
        ).mean()

    def q_loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = q_ts.apply_fn(params, replay_batch["state"])
        played_qs = jax.vmap(lambda heads_qs, i: heads_qs[i])(
            qs, replay_batch["action"]
        )
        return q_ts.loss_metric(td_targets, played_qs).mean()

    td_targets = agent_utils.bellman_target(
        gamma,
        v_ts.apply_fn(v_ts.target_params, replay_batch["next_state"]),
        replay_batch["reward"],
        replay_batch["terminal"],
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
class DQV(base.Agent):
    Q_model_def: agent_utils.ModelDefStore = field(kw_only=True)
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqn.DQN._make_train_state(self, self.V_model_def, True)
        self.models["Q"] = dqn.DQN._make_train_state(self, self.Q_model_def, False)
        dqn.DQN._set_exploration_fn(self)
        self.train_fn = train_dqv

    @property
    def act_selection_params(self) -> FrozenDict:
        return self.models["Q"].params

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def sync_weights(self):
        self.models["V"] = dqn.sync_weights(self.models["V"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        v_loss, self.models["V"], q_loss, self.models["Q"] = self.train_fn(
            self.gamma, *self.models.values(), experience_batch
        )
        return {"loss": {"V": v_loss, "Q": q_loss}}

    @property
    def reportable(self):
        return super().reportable + ("Q_model_def", "V_model_def")


@define
class MultiHeadEnsembleDQV(DQV):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = self.reassemble_V()

    def reassemble_V(self) -> ValueBasedTS:
        return self.models["V"].replace(
            loss_metric=lambda ys, xs: self.V_model_def.loss_fn(
                ys, xs, **self.V_model_def.loss_fn_params
            )
        )

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        return dqn.MultiHeadEnsembleDQN.train(self, experience_batch)
