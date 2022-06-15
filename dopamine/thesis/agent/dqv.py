from typing import Dict, Tuple, Union

import gin
import jax
import numpy as np
from attrs import define, field
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import custom_pytrees, types
from thesis.agent import base, dqn
from thesis.agent import utils as agent_utils

DQVModelTypes = Dict[
    str, Union[custom_pytrees.ValueBasedTS, custom_pytrees.ValueBasedTSEnsemble]
]


@jax.jit
def train_V(
    tr_state: custom_pytrees.ValueBasedTS,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    td_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def loss_fn(params, targets):
        s_t_estimates = agent_utils.batch_net_eval(tr_state.apply_fn, params, states)
        return jnp.mean(jax.vmap(tr_state.loss_metric)(targets, s_t_estimates))

    loss, grads = jax.value_and_grad(loss_fn)(tr_state.params, td_targets)
    return loss, tr_state.apply_gradients(grads=grads)


def train(
    experience_batch: Dict[str, np.ndarray],
    models: DQVModelTypes,
    gamma: float,
) -> Tuple[types.MetricsDict, DQVModelTypes]:
    td_targets = agent_utils.apply_td_loss(
        models["V"].s_tp1_fn, models["V"].target_params, experience_batch, gamma
    )
    v_loss, models["V"] = train_V(
        models["V"], experience_batch["state"], experience_batch["action"], td_targets
    )
    q_loss, models["Q"] = dqn.train_Q(
        models["Q"], experience_batch["state"], experience_batch["action"], td_targets
    )
    return {"loss": {"V": v_loss, "Q": q_loss}}, models


def train_ensembled(
    experience_batch: Dict[str, np.ndarray],
    models: DQVModelTypes,
    gamma: float,
) -> Tuple[types.MetricsDict, DQVModelTypes]:
    a_head = models["V"][0]
    td_targets = agent_utils.apply_td_loss(
        a_head.s_tp1_fn, a_head.target_params, experience_batch, gamma
    )
    v_losses_and_models = [
        train_V(
            head_state,
            experience_batch["state"],
            experience_batch["action"],
            td_targets,
        )
        for head_state in models["V"]
    ]
    v_losses, v_models = list(zip(*v_losses_and_models))
    models["V"] = custom_pytrees.ValueBasedTSEnsemble(v_models)
    q_loss, models["Q"] = dqn.train_Q(
        models["Q"], experience_batch["state"], experience_batch["action"], td_targets
    )
    return {"loss": {"V": jnp.array(v_losses).mean(), "Q": q_loss}}, models


@gin.configurable
@define
class DQV(base.Agent):
    Q_model_def: agent_utils.ModelDefStore = field(kw_only=True)
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = self._make_V_train_state(True)
        self.models["Q"] = agent_utils.build_TS(
            self.Q_model_def,
            self.rng,
            self.observation_shape,
            self.Q_model_def.net.apply,
            lambda x: x,  # placeholder, DQV Q's regresssion target == V's
            False,
        )
        self._set_exploration_fn()

    def _set_exploration_fn(self):
        dqn.DQN._set_exploration_fn(self)

    # DQV-Max also uses a V model, but the latter has no target network
    # in DQV-Max; differentiate on target_model
    def _make_V_train_state(self, target_model: bool) -> custom_pytrees.ValueBasedTS:
        return agent_utils.build_TS(
            self.V_model_def,
            self.rng,
            self.observation_shape,
            self.V_model_def.net.apply,
            lambda params, xs: agent_utils.batch_net_eval(
                self.V_model_def.net.apply, params, xs
            ),
            target_model,
        )

    @property
    def act_selection_params(self) -> frozen_dict.FrozenDict:
        return self.models["Q"].params

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def sync_weights(self):
        self.models["V"] = dqn.sync_weights(self.models["V"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models = train(experience_batch, self.models, self.gamma)
        return train_info

    @property
    def reportable(self):
        return super().reportable + ("Q_model_def", "V_model_def")


@gin.configurable
@define
class DQVEnsemble(DQV):
    def _make_V_train_state(
        self, target_model: bool
    ) -> custom_pytrees.ValueBasedTSEnsemble:
        return agent_utils.build_TS_ensemble(
            self.V_model_def,
            self.rng,
            self.observation_shape,
            lambda head, params, xs: self.V_model_def.net.apply(params, xs, head=head),
            lambda params, xs: agent_utils.batch_net_eval(
                self.V_model_def.net.apply, params, xs
            ).mean(axis=1),
            target_model,
        )

    def _set_exploration_fn(self):
        dqn.DQNEnsemble._set_exploration_fn(self)

    def sync_weights(self):
        self.models["V"] = dqn.sync_weights_ensemble(self.models["V"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models = train_ensembled(
            experience_batch, self.models, self.gamma
        )
        return train_info
