# import functools as ft
# import logging
# from typing import Dict, Tuple

# import gin
# import numpy as np
# from attrs import define, field
# from jax import numpy as jnp
# from jax import random as jrand
# from thesis import custom_pytrees, types
# from thesis.agent import dqn, dqv
# from thesis.agent import utils as agent_utils

# logger = logging.getLogger(__name__)


# @gin.configurable
# @define
# class BootstrappedDQVMax(dqn.BootstrappedDQN):
#     V_model_def: agent_utils.ModelDefStore = field(kw_only=True)
#     bootstrap_v_head_idx: int = field(init=None, default=None)

#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()
#         self.models["V"] = dqv.BootstrappedDQV._make_V_train_state(self, False)

#     @property
#     def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
#         return {"loss": {"V": jnp.zeros(()), "Q": jnp.zeros(())}}

#     def on_episode_start(self, mode: str):
#         q_idx, v_idx = jrand.randint(next(self.rng), (2,), 0, len(self.models["Q"]))
#         self.bootstrap_head_idx, self.bootstrap_v_head_idx = q_idx, v_idx
#         self.policy_evaluator.model_call = self.models["Q"][q_idx].apply_fn
#         logger.debug(f"***Next {mode} episode head index: V {v_idx} Q {q_idx}")

#     def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
#         train_info, models = train(
#             experience_batch,
#             {
#                 "Q": self.models["Q"][self.bootstrap_head_idx],
#                 "V": self.models["V"][self.bootstrap_v_head_idx],
#             },
#             self.gamma,
#         )
#         self.models["Q"][self.bootstrap_head_idx] = models["Q"]
#         self.models["V"][self.bootstrap_v_head_idx] = models["V"]
#         return train_info


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


# @gin.configurable
# @define
# class BootstrappedDQV(dqv.DQV):
#     ensemble_td_target: bool = False
#     bootstrap_head_idx: int = field(init=None, default=None)

#     def _make_V_train_state(
#         self, target_model: bool
#     ) -> custom_pytrees.ValueBasedTSEnsemble:
#         if self.ensemble_td_target:
#             td_target_fn = lambda _: lambda params, xs: (
#                 agent_utils.batch_net_eval(self.V_model_def.net.apply, params, xs).mean(
#                     1
#                 )
#             )
#         else:
#             td_target_fn = lambda head: lambda params, xs: agent_utils.batch_net_eval(
#                 ft.partial(self.V_model_def.net.apply, head=head), params, xs
#             )
#         return agent_utils.build_TS_ensemble(
#             self.V_model_def,
#             self.rng,
#             self.observation_shape,
#             lambda head: lambda params, xs: self.V_model_def.net.apply(
#                 params, xs, head=head
#             ),
#             td_target_fn,
#             target_model,
#         )

#     def on_episode_start(self, mode: str):
#         idx = jrand.randint(next(self.rng), (), 0, len(self.models["V"]))

#         self.bootstrap_head_idx = idx
#         logger.debug(f"***Next {mode} episode head index: {idx}")

#     def sync_weights(self):
#         self.models["V"] = dqn.sync_weights_ensemble(self.models["V"])

#     def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
#         train_info, models = train(
#             experience_batch,
#             {"Q": self.models["Q"], "V": self.models["V"][self.bootstrap_head_idx]},
#             self.gamma,
#         )
#         self.models["Q"] = models["Q"]
#         self.models["V"][self.bootstrap_head_idx] = models["V"]
#         return train_info

#     @property
#     def reportable(self) -> Tuple[str]:
#         return super().reportable + ("ensemble_td_target",)


# def train_ensembled(
#     experience_batch: Dict[str, np.ndarray],
#     models: DQVModelTypes,
#     gamma: float,
#     rng: custom_pytrees.PRNGKeyWrap,
# ) -> Tuple[types.MetricsDict, DQVModelTypes]:
#     a_head = models["V"][jrand.randint(next(rng), (), 0, len(models["V"]))]
#     td_targets = agent_utils.apply_td_loss(
#         a_head.s_tp1_fn, a_head.target_params, experience_batch, gamma
#     )
#     v_losses_and_models = [
#         train_V(
#             head_state,
#             experience_batch["state"],
#             experience_batch["action"],
#             td_targets,
#         )
#         for head_state in models["V"]
#     ]
#     v_losses, v_models = list(zip(*v_losses_and_models))
#     models["V"] = custom_pytrees.ValueBasedTSEnsemble(v_models)
#     q_loss, models["Q"] = dqn.train_Q(
#         models["Q"], experience_batch["state"], experience_batch["action"], td_targets
#     )
#     return {"loss": {"V": jnp.array(v_losses).mean(), "Q": q_loss}}, models


# DQNNNNNN

# @gin.configurable
# @define
# class BootstrappedDQN(DQN):
#     ensemble_td_target: bool = False
#     bootstrap_head_idx: int = field(init=None, default=None)

#     def _make_Q_train_state(self) -> custom_pytrees.ValueBasedTSEnsemble:
#         if self.ensemble_td_target:
#             td_target_fn = lambda _: lambda params, xs: (
#                 agent_utils.batch_net_eval(self.Q_model_def.net.apply, params, xs).mean(
#                     1
#                 )
#             ).max(1)
#         else:
#             td_target_fn = lambda head: lambda params, xs: agent_utils.batch_net_eval(
#                 ft.partial(self.Q_model_def.net.apply, head=head), params, xs
#             ).max(1)
#         return agent_utils.build_TS_ensemble(
#             self.Q_model_def,
#             self.rng,
#             self.observation_shape,
#             lambda head: lambda params, xs: self.Q_model_def.net.apply(
#                 params, xs, head=head
#             ),
#             td_target_fn,
#             True,
#         )

#     def _set_exploration_fn(self):
#         pass

#     # following the bootstrapped-dqn scheme
#     # (https://arxiv.org/abs/1602.04621), one head is picked and used for
#     # control for a whole episode
#     # TODO should this be the same for evaluation?
#     def on_episode_start(self, mode: str):
#         idx = jrand.randint(next(self.rng), (), 0, len(self.models["Q"]))
#         self.bootstrap_head_idx = idx
#         self.policy_evaluator.model_call = self.models["Q"][idx].apply_fn
#         logger.debug(f"***Next {mode} episode head index: {idx}")

#     @property
#     def act_selection_params(self) -> FrozenDict:
#         return self.models["Q"][self.bootstrap_head_idx].params

#     def sync_weights(self):
#         self.models["Q"] = sync_weights_ensemble(self.models["Q"])

#     def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
#         train_info, self.models["Q"][self.bootstrap_head_idx] = train(
#             experience_batch, self.models["Q"][self.bootstrap_head_idx], self.gamma
#         )
#         return train_info

#     @property
#     def reportable(self) -> Tuple[str]:
#         return super().reportable + ("ensemble_td_target",)


# def train_ensembled(
#     experience_batch: Dict[str, np.ndarray],
#     q_model: custom_pytrees.ValueBasedTSEnsemble,
#     gamma: float,
#     rng: custom_pytrees.PRNGKeyWrap,
# ) -> Tuple[types.MetricsDict, custom_pytrees.ValueBasedTSEnsemble]:
#     # pick random prediction head
#     a_head = q_model[jrand.randint(next(rng), (), 0, len(q_model))]
#     td_targets = agent_utils.apply_td_loss(
#         a_head.s_tp1_fn, a_head.target_params, experience_batch, gamma
#     )
#     q_losses_and_models = [
#         train_Q(
#             q_head,
#             experience_batch["state"],
#             experience_batch["action"],
#             td_targets,
#         )
#         for q_head in q_model
#     ]
#     q_losses, q_model = list(zip(*q_losses_and_models))
#     return {
#         "loss": {"Q": jnp.array(q_losses).mean()}
#     }, custom_pytrees.ValueBasedTSEnsemble(q_model)
