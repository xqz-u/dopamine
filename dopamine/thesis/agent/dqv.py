import functools as ft
import logging
from typing import Dict, Tuple, Union

import gin
import jax
import numpy as np
from attrs import define, field
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, types
from thesis.agent import base, dqn
from thesis.agent import utils as agent_utils

DQVModelTypes = Dict[
    str, Union[custom_pytrees.ValueBasedTS, custom_pytrees.ValueBasedTSEnsemble]
]


logger = logging.getLogger(__name__)


# NOTE V is multihead so operations are batched, Q is a normal network
@ft.partial(jax.jit, static_argnums=(0,))
def train_DQV_multihead_tiny(
    gamma: float,
    v_ts: custom_pytrees.ValueBasedTS,
    q_ts: custom_pytrees.ValueBasedTS,
    replay_batch: Dict[str, np.ndarray],
) -> Tuple[Tuple[jnp.ndarray], Tuple[custom_pytrees.ValueBasedTS]]:
    def v_loss_fn(params: FrozenDict) -> jnp.ndarray:
        vs = v_ts.apply_fn(params, replay_batch["state"])
        # loss on each dimension - only 1 - for each head
        v_heads_loss = v_ts.loss_metric(td_targets, vs)
        # mean loss on sample among heads, then mean across samples
        return v_heads_loss.mean()

    def q_loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = jax.vmap(lambda s: q_ts.apply_fn(params, s))(replay_batch["state"])
        played_qs = jax.vmap(lambda heads_qs, i: heads_qs[i])(
            qs, replay_batch["action"]
        )
        return jax.vmap(q_ts.loss_metric)(td_targets, played_qs).mean()

    td_targets = jax.lax.stop_gradient(
        agent_utils.bellman_target(
            gamma,
            v_ts.apply_fn(v_ts.target_params, replay_batch["next_state"]),
            jnp.expand_dims(replay_batch["reward"], 1),
            jnp.expand_dims(replay_batch["terminal"], 1),
        )
    )
    v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_ts.params)
    q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_ts.params)
    return (v_loss, q_loss), (
        v_ts.apply_gradients(grads=v_grads),
        q_ts.apply_gradients(grads=q_grads),
    )


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
    def act_selection_params(self) -> FrozenDict:
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


class MultiHeadEnsembleDQVTiny(DQV):
    def _make_V_train_state(self, target_model: bool) -> custom_pytrees.ValueBasedTS:
        return agent_utils.build_TS(
            self.V_model_def,
            self.rng,
            self.observation_shape,
            lambda params, xs: jax.vmap(
                lambda x: self.V_model_def.net.apply(params, x)
            )(xs),
            None,
            target_model,
        )

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        (v_loss, q_loss), (v_ts, q_ts) = train_DQV_multihead_tiny(
            self.gamma, self.models["V"], self.models["Q"], experience_batch
        )
        self.models["V"] = v_ts
        self.models["Q"] = q_ts
        return {"loss": {"V": v_loss, "Q": q_loss}}


@gin.configurable
@define
class BootstrappedDQV(DQV):
    ensemble_td_target: bool = False
    bootstrap_head_idx: int = field(init=None, default=None)

    def _make_V_train_state(
        self, target_model: bool
    ) -> custom_pytrees.ValueBasedTSEnsemble:
        if self.ensemble_td_target:
            td_target_fn = lambda _: lambda params, xs: (
                agent_utils.batch_net_eval(self.V_model_def.net.apply, params, xs).mean(
                    1
                )
            )
        else:
            td_target_fn = lambda head: lambda params, xs: agent_utils.batch_net_eval(
                ft.partial(self.V_model_def.net.apply, head=head), params, xs
            )
        return agent_utils.build_TS_ensemble(
            self.V_model_def,
            self.rng,
            self.observation_shape,
            lambda head: lambda params, xs: self.V_model_def.net.apply(
                params, xs, head=head
            ),
            td_target_fn,
            target_model,
        )

    def on_episode_start(self, mode: str):
        idx = jrand.randint(next(self.rng), (), 0, len(self.models["V"]))

        self.bootstrap_head_idx = idx
        logger.debug(f"***Next {mode} episode head index: {idx}")

    def sync_weights(self):
        self.models["V"] = dqn.sync_weights_ensemble(self.models["V"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, models = train(
            experience_batch,
            {"Q": self.models["Q"], "V": self.models["V"][self.bootstrap_head_idx]},
            self.gamma,
        )
        self.models["Q"] = models["Q"]
        self.models["V"][self.bootstrap_head_idx] = models["V"]
        return train_info

    @property
    def reportable(self) -> Tuple[str]:
        return super().reportable + ("ensemble_td_target",)


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
