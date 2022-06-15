from abc import ABC, abstractmethod
from typing import Tuple

import gin
from attrs import define, field
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import custom_pytrees, types

# NOTE always pass an RNG: it has shared state maintained by an Agent,
# and having a reference here too implies more error-prone bookkeeping


# model_call is set by each agent once its model and train states are
# created
@gin.configurable
@define
class PolicyEvaluator(ABC):
    num_actions: int
    model_call: types.ModuleCall = field(init=False, default=None)

    @abstractmethod
    def __call__(
        self,
        rng: custom_pytrees.PRNGKeyWrap,
        mode: str,
        params: frozen_dict.FrozenDict,
        state: jnp.ndarray,
        **kwargs
    ) -> types.PolicyEvalInfo:
        ...

    @property
    def reportable(self) -> Tuple[str]:
        return [a.name for a in self.__attrs_attrs__ if a.name != "model_call"]
