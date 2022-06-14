import functools as ft

import gin
import jax
from attrs import define
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import custom_pytrees, types
from thesis.exploration import egreedy


# taken from dopamine.jax.agents.dqn.dqn_agent
@ft.partial(jax.jit, static_argnums=(0, 1, 2))
def linearly_decaying_epsilon(
    decay_period: int, warmup_steps: int, epsilon_target: float, training_steps: int
) -> jnp.ndarray:
    steps_left = decay_period + warmup_steps - training_steps
    bonus = (1.0 - epsilon_target) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0.0, 1.0 - epsilon_target)
    return epsilon_target + bonus


@gin.configurable
@define
class EgreedyLinearDecay(egreedy.Egreedy):
    decay_period: int = 250000
    warmup_steps: int = 500

    def __call__(
        self,
        rng: custom_pytrees.PRNGKeyWrap,
        mode: str,
        params: frozen_dict.FrozenDict,
        state: jnp.ndarray,
        training_steps: int = None,
    ) -> types.PolicyEvalInfo:
        return egreedy.egreedy(
            self.model_call,
            self.num_actions,
            linearly_decaying_epsilon(
                self.decay_period,
                self.warmup_steps,
                self._epsilon_train,
                training_steps,
            )
            if mode == "train"
            else self.epsilon_eval,
            rng,
            params,
            state,
        )
