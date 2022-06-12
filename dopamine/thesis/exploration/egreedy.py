import functools as ft
from typing import Union

import gin
import jax
from attrs import define
from flax.core import frozen_dict
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, types
from thesis.exploration import base


# always perform a state evaluation to return the maximum state Q-value
# as part of the running metrics
@ft.partial(jax.jit, static_argnums=(0, 1))
def egreedy(
    model_call: types.ModuleCall,
    num_actions: int,
    epsilon: Union[jnp.ndarray, float],
    rng: custom_pytrees.PRNGKeyWrap,
    params: frozen_dict.FrozenDict,
    state: jnp.ndarray,
) -> types.PolicyEvalInfo:
    state_qs = model_call(params, state)
    action = jnp.where(
        jrand.uniform(next(rng)) <= epsilon,
        jrand.randint(next(rng), (), 0, num_actions),
        state_qs.argmax(),
    )
    return rng, action, {"max_q": state_qs.max()}


@gin.configurable
@define
class Egreedy(base.PolicyEvaluator):
    epsilon_train: float = 0.01
    epsilon_eval: float = 0.001

    def __call__(
        self,
        rng: custom_pytrees.PRNGKeyWrap,
        mode: str,
        params: frozen_dict.FrozenDict,
        state: jnp.ndarray,
        **_,
    ) -> types.PolicyEvalInfo:
        return egreedy(
            self.model_call,
            self.num_actions,
            getattr(self, f"epsilon_{mode}"),
            rng,
            params,
            state,
        )
