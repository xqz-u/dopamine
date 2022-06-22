import logging
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import gin
import optax
from flax import struct
from flax.core import frozen_dict
from flax.training import train_state
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util

logger = logging.getLogger(__name__)


# NOTE circular import; but:
# s_tp1_fn: thesis.types.ModuleCall
# loss_metric: thesis.types.LossMetric
class ValueBasedTS(train_state.TrainState):
    s_tp1_fn: Callable[
        [frozen_dict.FrozenDict, jnp.ndarray], jnp.ndarray
    ] = struct.field(pytree_node=False)
    loss_metric: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = struct.field(
        pytree_node=False
    )
    target_params: frozen_dict.FrozenDict

    @property
    def serializable(self) -> Dict[str, Union[frozen_dict.FrozenDict, optax.OptState]]:
        return {"params": self.params, "opt_state": self.opt_state}


# iterable version of ValueBasedTS, stores multiple train states
# and is used like a sequence
# NOTE could be made a PyTreeNode if passed to jax transformed fns
@dataclass
class ValueBasedTSEnsemble:
    TSS: Tuple[ValueBasedTS, ...]

    def __getitem__(self, idx: int) -> ValueBasedTS:
        return self.TSS[idx]

    def __len__(self) -> int:
        return len(self.TSS)

    @property
    def serializable(
        self,
    ) -> Dict[str, Dict[str, Union[frozen_dict.FrozenDict, optax.OptState]]]:
        return {head: head_TS.serializable for head, head_TS in enumerate(self.TSS)}


@gin.configurable
@tree_util.register_pytree_node_class
@dataclass
# NOTE when key and n_splits are passed together with seed, the first
# must correspond to real KeyArray obtained from seed after n_splits
# in order  to be reproducible
class PRNGKeyWrap:
    seed: int = 42
    key: jrand.KeyArray = None
    n_splits: int = 0

    def __post_init__(self):
        if self.key is None:
            self.reset()

    def __next__(self) -> jrand.KeyArray:
        self.key, sk = jrand.split(self.key)
        self.n_splits += 1
        return sk

    def reset(self):
        logger.debug(f"Reset {self}")
        self.key = jrand.PRNGKey(self.seed)
        self.n_splits = 0
        logger.debug(f"After reset: {self}")

    def tree_flatten(self) -> Tuple[Tuple[int, jrand.KeyArray, int], None]:
        return ((self.seed, self.key, self.n_splits), None)

    @classmethod
    def tree_unflatten(cls, _, leaves):
        return cls(*leaves)

    @classmethod
    def from_dict(cls, serialized_rng: dict) -> "PRNGKeyWrap":
        return cls(**serialized_rng)

    @property
    def reportable(self) -> Tuple[str]:
        return ("seed",)

    @property
    def serializable(self) -> Dict[str, Union[jnp.ndarray, int]]:
        # cast: when going through a jitted function, a PyTree's
        # attributes are concretized/traced and lose original type
        return {
            "key": jnp.array(self.key),
            "seed": int(self.seed),
            "n_splits": int(self.n_splits),
        }

    def __repr__(self) -> str:
        return f"<seed: {self.seed} #splits: {self.n_splits} key: {self.key}>"
