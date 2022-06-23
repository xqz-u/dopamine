import functools as ft
import logging
from typing import Dict, Tuple

import gin
import jax
import numpy as np
from attrs import define, field
from flax.core import frozen_dict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, types
from thesis.agent import base
from thesis.agent import utils as agent_utils

logger = logging.getLogger(__name__)


@jax.jit
def train_Q(
    tr_state: custom_pytrees.ValueBasedTS,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    td_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def loss_fn(params, targets):
        s_t_estimates = agent_utils.batch_net_eval(tr_state.apply_fn, params, states)
        s_t_estimates = jax.vmap(lambda q, a: q[a])(s_t_estimates, actions)

        return jnp.mean(jax.vmap(tr_state.loss_metric)(targets, s_t_estimates))

    loss, grads = jax.value_and_grad(loss_fn)(tr_state.params, td_targets)
    return loss, tr_state.apply_gradients(grads=grads)


def train(
    experience_batch: Dict[str, np.ndarray],
    q_model: custom_pytrees.ValueBasedTS,
    gamma: float,
) -> Tuple[types.MetricsDict, custom_pytrees.ValueBasedTS]:
    td_targets = agent_utils.apply_td_loss(
        q_model.s_tp1_fn, q_model.target_params, experience_batch, gamma
    )
    q_loss, q_model = train_Q(
        q_model, experience_batch["state"], experience_batch["action"], td_targets
    )
    return {"loss": {"Q": q_loss}}, q_model


def train_ensembled(
    experience_batch: Dict[str, np.ndarray],
    q_model: custom_pytrees.ValueBasedTSEnsemble,
    gamma: float,
    rng: custom_pytrees.PRNGKeyWrap,
) -> Tuple[types.MetricsDict, custom_pytrees.ValueBasedTSEnsemble]:
    # pick random prediction head
    a_head = q_model[jrand.randint(next(rng), (), 0, len(q_model))]
    td_targets = agent_utils.apply_td_loss(
        a_head.s_tp1_fn, a_head.target_params, experience_batch, gamma
    )
    q_losses_and_models = [
        train_Q(
            q_head,
            experience_batch["state"],
            experience_batch["action"],
            td_targets,
        )
        for q_head in q_model
    ]
    q_losses, q_model = list(zip(*q_losses_and_models))
    return {
        "loss": {"Q": jnp.array(q_losses).mean()}
    }, custom_pytrees.ValueBasedTSEnsemble(q_model)


# cannot simply assign since flax.train_state.TrainState is a dataclass
# with frozen=True (necessary for automatic pure transformations as
# implemented when passing a PyTree to a jax.jitted function)
def sync_weights(model: custom_pytrees.ValueBasedTS) -> custom_pytrees.ValueBasedTS:
    return model.replace(target_params=model.params)


def sync_weights_ensemble(
    model: custom_pytrees.ValueBasedTSEnsemble,
) -> custom_pytrees.ValueBasedTSEnsemble:
    return custom_pytrees.ValueBasedTSEnsemble([sync_weights(head) for head in model])


# `kw_only=True` is required when mandatory attributes follow ones with
# defaults, as it is the case in my agents' inheritance chain
@gin.configurable
@define
class DQN(base.Agent):
    Q_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["Q"] = self._make_Q_train_state()
        self._set_exploration_fn()

    def _make_Q_train_state(self) -> custom_pytrees.ValueBasedTS:
        return agent_utils.build_TS(
            self.Q_model_def,
            self.rng,
            self.observation_shape,
            self.Q_model_def.net.apply,
            lambda params, xs: agent_utils.batch_net_eval(
                self.Q_model_def.net.apply, params, xs
            ).max(
                1  # max across Q-values (col index == action)
            ),
            True,
        )

    def _set_exploration_fn(self):
        self.policy_evaluator.model_call = self.models["Q"].apply_fn

    @property
    def act_selection_params(self) -> frozen_dict.FrozenDict:
        return self.models["Q"].params

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"Q": jnp.zeros(())}}

    def sync_weights(self):
        self.models["Q"] = sync_weights(self.models["Q"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models["Q"] = train(
            experience_batch, self.models["Q"], self.gamma
        )
        return train_info

    @property
    def reportable(self) -> Tuple[str]:
        return super().reportable + ("Q_model_def",)


@gin.configurable
@define
class BootstrappedDQN(DQN):
    ensemble_td_target: bool = False
    bootstrap_head_idx: int = field(init=None, default=None)

    def _make_Q_train_state(self) -> custom_pytrees.ValueBasedTSEnsemble:
        td_target_fn = lambda head: lambda params, xs: (
            agent_utils.batch_net_eval(self.Q_model_def.net.apply, params, xs).mean(
                axis=1
            )
            if self.ensemble_td_target
            else agent_utils.batch_net_eval(
                ft.partial(self.Q_model_def.net.apply, head=head), params, xs
            )
        ).max(axis=1)
        return agent_utils.build_TS_ensemble(
            self.Q_model_def,
            self.rng,
            self.observation_shape,
            lambda head: lambda params, xs: self.Q_model_def.net.apply(
                params, xs, head=head
            ),
            td_target_fn,
            True,
        )

    def _set_exploration_fn(self):
        pass

    def on_episode_start(self, mode: str):
        idx = jrand.randint(next(self.rng), (), 0, len(self.models["Q"]))
        self.bootstrap_head_idx = idx
        self.policy_evaluator.model_call = self.models["Q"][idx].apply_fn
        logger.debug(
            f"Next {mode} episode head index: {idx} explore id: {id(self.policy_evaluator.model_call)}"
        )

    @property
    def act_selection_params(self) -> frozen_dict.FrozenDict:
        return self.models["Q"][self.bootstrap_head_idx].params

    def sync_weights(self):
        self.models["Q"][self.bootstrap_head_idx] = sync_weights(
            self.models["Q"][self.bootstrap_head_idx]
        )

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models["Q"][self.bootstrap_head_idx] = train(
            experience_batch, self.models["Q"][self.bootstrap_head_idx], self.gamma
        )
        return train_info

    @property
    def reportable(self) -> Tuple[str]:
        return super().reportable + ("ensemble_td_target",)


class DQNEnsemble(DQN):
    ...


# self.policy_evaluator.model_call = lambda params, x: self.Q_model_def.net.apply(
#     params, x
# ).mean(axis=0)
# self.models["Q"] = sync_weights_ensemble(self.models["Q"])
# def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
#     train_info, self.models["Q"] = train_ensembled(
#         experience_batch, self.models["Q"], self.gamma, self.rng
#     )
#     return train_info
