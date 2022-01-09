from dataclasses import dataclass
from typing import Sequence

from jax import numpy as jnp
from jax import random as jrand


# NOTE desired syntax would be smth like:
# with mykeywrap(key) as nk:
#     jrand.uniform(nk, ...)
#     jrand.randint(nk, ...)
# where each nk is substituted for the correct new key
@dataclass
class PRNGKeyWrap:
    seed: int = 42
    rng: jrand.KeyArray = None
    _prev_rng: jrand.KeyArray = None  # NOTE useful?
    n_splits: int = 0

    def __post_init__(self):
        if self.rng is None:
            self.rng = jrand.PRNGKey(self.seed)
        else:
            self.seed = int(self.rng[1])
        self._prev_rng = self.rng

    def __next__(self) -> jrand.KeyArray:
        self.rng, sk = jrand.split(self.rng)
        self._prev_rng = sk
        self.n_splits += 1
        return sk

    def __iter__(self):
        return self


# PRNGKeyWrap(seed=0)
# PRNGKeyWrap()
# PRNGKeyWrap(rng=jrand.PRNGKey(5))
# PRNGKeyWrap(seed=1, rng=jrand.PRNGKey(4))

# x = PRNGKeyWrap()
# for i in range(5):
#     print(next(x))

# y = next(x)
# assert y == x._prev_rng


def force_devicearray_split(
    key: jrand.KeyArray, n: int = 2
) -> Sequence[jrand.KeyArray]:
    splits = jrand.split(key, n)
    return [jnp.array(k) for k in splits]
