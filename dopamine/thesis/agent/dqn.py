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
from thesis.agent import base
from thesis.agent import utils as agent_utils
from thesis.custom_pytrees import ValueBasedTS

logger = logging.getLogger(__name__)


@ft.partial(jax.jit, static_argnums=(0,))
def train_q(
    gamma: float, ts: ValueBasedTS, replay_batch: Dict[str, np.ndarray]
) -> Tuple[jnp.ndarray, ValueBasedTS]:
    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        qs = ts.apply_fn(params, replay_batch["state"])
        played_qs = jax.vmap(lambda q, a: q[a])(qs, replay_batch["action"])
        return ts.loss_metric(td_targets, played_qs).mean()

    td_targets = agent_utils.bellman_target(
        gamma,
        ts.apply_fn(ts.target_params, replay_batch["next_state"]).max(1),
        replay_batch["reward"],
        replay_batch["terminal"],
    )
    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
    return loss, ts.apply_gradients(grads=grads)


# TODO as field like DQN.train_fn?
# cannot simply assign since flax.train_state.TrainState is a dataclass
# with frozen=True (necessary for automatic pure transformations as
# implemented when passing a PyTree to a jax.jitted function)
def sync_weights(model: ValueBasedTS) -> ValueBasedTS:
    return model.replace(target_params=model.params)


# `kw_only=True` is required when mandatory attributes follow ones with
# defaults, as it is the case in my agents' inheritance chain
@gin.configurable
@define
class DQN(base.Agent):
    Q_model_def: agent_utils.ModelDefStore = field(kw_only=True)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["Q"] = self._make_train_state(self.Q_model_def, True)
        self._set_exploration_fn()
        self.train_fn = train_q

    # DQV uses Q-network with same structure, but with only one set of
    # parameters
    def _make_train_state(
        self, model_def: agent_utils.ModelDefStore, target_model: bool
    ) -> ValueBasedTS:
        return agent_utils.build_TS(
            model_def,
            self.rng,
            self.observation_shape,
            lambda params, xs: jax.vmap(lambda s: model_def.net.apply(params, s))(xs),
            lambda ys, xs: jax.vmap(
                lambda y, x: model_def.loss_fn(y, x, **model_def.loss_fn_params)
            )(ys, xs),
            target_model,
        )

    def _set_exploration_fn(self):
        self.policy_evaluator.model_call = lambda params, s: self.Q_model_def.net.apply(
            params, s
        )

    @property
    def act_selection_params(self) -> FrozenDict:
        return self.models["Q"].params

    @property
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {"loss": {"Q": jnp.zeros(())}}

    def sync_weights(self):
        self.models["Q"] = sync_weights(self.models["Q"])

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        train_info, self.models["Q"] = self.train_fn(
            self.gamma, self.models["Q"], experience_batch
        )
        return {"loss": {"Q": train_info}}

    @property
    def reportable(self) -> Tuple[str]:
        return super().reportable + ("Q_model_def",)


@define
class MultiHeadEnsembleDQN(DQN):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.models["Q"] = self.reassemble_Q()

    def reassemble_Q(self) -> ValueBasedTS:
        reshape_spec = (
            self.policy_evaluator.num_actions,
            self.Q_model_def.info["n_heads"],
        )
        base_model_explore = self.policy_evaluator.model_call
        self.policy_evaluator.model_call = (
            lambda params, s: base_model_explore(params, s)
            .reshape(reshape_spec)
            .mean(1)
        )
        ts = self.models["Q"]
        return ts.replace(
            apply_fn=lambda params, xs: ts.apply_fn(params, xs).reshape(
                (-1, *reshape_spec)
            ),
            loss_metric=lambda ys, xs: self.Q_model_def.loss_fn(
                ys, xs, **self.Q_model_def.loss_fn_params
            ),
        )

    def train(self, experience_batch: Dict[str, np.ndarray]) -> types.MetricsDict:
        for k in ["reward", "terminal"]:
            experience_batch[k] = jnp.expand_dims(experience_batch[k], 1)
        return super(type(self), self).train(experience_batch)
