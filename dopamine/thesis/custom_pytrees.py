from dataclasses import dataclass
from typing import Dict, Union

import optax
from dopamine.jax import losses
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util

from thesis import networks


@tree_util.register_pytree_node_class
@dataclass
class NetworkOptimWrap:
    params: Union[FrozenDict, Dict[str, FrozenDict]] = None
    optim_state: optax.OptState = None
    net: nn.Module = networks.Sequential
    optim: optax.GradientTransformation = optax.sgd
    loss_metric: callable = losses.mse_loss

    def tree_flatten(self) -> tuple:
        return (
            (self.params, self.optim_state),
            (self.net, self.optim, self.loss_metric),
        )

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(*leaves, *treedef)

    @property
    def checkpointable_elements(self) -> dict:
        return {"params": self.params, "optim_state": self.optim_state}


# NOTE if using a dataclass, this happens:
# TypeError: __init__() takes from 1 to 4 positional arguments but 5 were given
@tree_util.register_pytree_node_class
@dataclass
class PRNGKeyWrap:
    def __init__(
        self,
        seed: int = 42,
        key: jrand.KeyArray = None,
        n_splits: int = 0,
        _stop_seed_assign=None,
    ):
        self.key = key
        self.seed = seed
        self._stop_seed_assign = _stop_seed_assign
        self.n_splits = n_splits
        if self.key is None:
            self.key = jrand.PRNGKey(self.seed)
        # NOTE _stop_seed_assign exists because the next statement
        # would otherwise be executed on each call to next() because
        # of how custom PyTrees work
        elif self._stop_seed_assign is None:
            self.seed = self.key[1].astype(int)

    def __next__(self) -> jrand.KeyArray:
        self.key, sk = jrand.split(self.key)
        self.n_splits += 1
        if self._stop_seed_assign is None:
            self._stop_seed_assign = 1
        return sk

    def __repr__(self) -> str:
        return f"<seed: {self.seed} #splits: {self.n_splits} key: {self.key}>"

    def tree_flatten(self) -> tuple:
        return ((self.seed, self.key, self.n_splits, self._stop_seed_assign), None)

    @classmethod
    def tree_unflatten(cls, _, leaves):
        return cls(*leaves)

    @classmethod
    def from_dict(cls, serialized_rng: dict):
        return cls(**serialized_rng)

    @property
    def checkpointable_elements(self) -> dict:
        # cast: when going through a jitted function, a PyTree's
        # attributes are concretized/traced and lose original type
        return {
            "key": jnp.array(self.key),
            "seed": int(self.seed),
            "n_splits": int(self.n_splits),
        }
