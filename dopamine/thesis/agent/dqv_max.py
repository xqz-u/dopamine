import logging
from typing import Dict, Tuple

import gin
import numpy as np
from attrs import define, field
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, types
from thesis.agent import dqn, dqv
from thesis.agent import utils as agent_utils

logger = logging.getLogger(__name__)


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


@gin.configurable
@define
class BootstrappedDQVMax(dqn.BootstrappedDQN):
    V_model_def: agent_utils.ModelDefStore = field(kw_only=True)
    bootstrap_v_head_idx: int = field(init=None, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["V"] = dqv.BootstrappedDQV._make_V_train_state(self, False)

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

    def on_episode_start(self, mode: str):
        q_idx, v_idx = jrand.randint(next(self.rng), (2,), 0, len(self.models["Q"]))
        self.bootstrap_head_idx, self.bootstrap_v_head_idx = q_idx, v_idx
        self.policy_evaluator.model_call = self.models["Q"][q_idx].apply_fn
        logger.debug(f"***Next {mode} episode head index: V {v_idx} Q {q_idx}")

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, models = train(
            experience_batch,
            {
                "Q": self.models["Q"][self.bootstrap_head_idx],
                "V": self.models["V"][self.bootstrap_v_head_idx],
            },
            self.gamma,
        )
        self.models["Q"][self.bootstrap_head_idx] = models["Q"]
        self.models["V"][self.bootstrap_v_head_idx] = models["V"]
        return train_info


class DQVMaxEnsemble:
    ...


# def train_ensembled(
#     experience_batch: Dict[str, np.ndarray],
#     models: dqv.DQVModelTypes,
#     gamma: float,
#     rng: custom_pytrees.PRNGKeyWrap,
# ) -> Tuple[types.MetricsDict, dqv.DQVModelTypes]:
#     # NOTE assumes v and q model have same number of heads
#     q_i, v_i = jrand.randint(next(rng), (2,), 0, len(models["V"]))
#     v_head, q_head = models["V"][v_i], models["Q"][q_i]
#     v_td_targets = agent_utils.apply_td_loss(
#         q_head.s_tp1_fn, q_head.target_params, experience_batch, gamma
#     )
#     q_td_targets = agent_utils.apply_td_loss(
#         v_head.s_tp1_fn, v_head.params, experience_batch, gamma
#     )
#     v_losses_and_models, q_losses_and_models = [
#         [
#             train_fn(
#                 head_state,
#                 experience_batch["state"],
#                 experience_batch["action"],
#                 td_targets,
#             )
#             for head_state in models[model_name]
#         ]
#         for train_fn, model_name, td_targets in (
#             (dqv.train_V, "V", v_td_targets),
#             (dqn.train_Q, "Q", q_td_targets),
#         )
#     ]
#     v_losses, v_model = list(zip(*v_losses_and_models))
#     q_losses, q_model = list(zip(*q_losses_and_models))
#     return {
#         "loss": {"V": jnp.array(v_losses).mean(), "Q": jnp.array(q_losses).mean()}
#     }, {
#         "V": custom_pytrees.ValueBasedTSEnsemble(v_model),
#         "Q": custom_pytrees.ValueBasedTSEnsemble(q_model),
#     }
