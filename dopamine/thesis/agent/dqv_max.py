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
def train_DQVMax_multihead(
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
    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_ts.params)
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_ts.params)
    return (
        v_loss,
        v_ts.apply_gradients(grads=v_grads),
        q_loss,
        q_ts.apply_gradients(grads=q_grads),
    )


# TODO rewrite, also for dqv and dqn
def train(
    experience_batch: Dict[str, np.ndarray],
    models: dqv.DQVModelTypes,
    gamma: float,
) -> Tuple[types.MetricsDict, dqv.DQVModelTypes]:
    v_td_targets = agent_utils.apply_td_loss(
        models["Q"].s_tp1_fn, models["Q"].target_params, experience_batch, gamma
    )
    q_td_targets = agent_utils.apply_td_loss(
        models["V"].s_tp1_fn, models["V"].params, experience_batch, gamma
    )
    v_loss, models["V"] = dqv.train_V(
        models["V"], experience_batch["state"], experience_batch["action"], v_td_targets
    )
    q_loss, models["Q"] = dqn.train_Q(
        models["Q"], experience_batch["state"], experience_batch["action"], q_td_targets
    )
    return {"loss": {"V": v_loss, "Q": q_loss}}, models


@gin.configurable
@define
class DQVMax(dqn.DQN):
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqv.DQV._make_V_train_state(self, False)

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models = train(experience_batch, self.models, self.gamma)
        return train_info

    @property
    def reportable(self):
        return super().reportable + ("V_model_def",)


@define
class MultiHeadEnsembleDQVMax(dqn.MultiHeadEnsembleDQN):
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqv.MultiHeadEnsembleDQVTiny._make_V_train_state(self, False)

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        v_loss, self.models["V"], q_loss, self.models["Q"] = train_DQVMax_multihead(
            self.gamma, *self.models.values(), experience_batch
        )
        return {"loss": {"V": v_loss, "Q": q_loss}}

    @property
    def reportable(self):
        return super().reportable + ("V_model_def",)
