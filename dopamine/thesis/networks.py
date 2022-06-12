from typing import Optional, Tuple, Union

import gin
import numpy as np
from dopamine.jax.networks import NatureDQNNetwork
from flax import linen as nn
from jax import numpy as jnp


# NOTE passing min_ and max_ vals as tuple keeps the model hashable,
# necessary for jitted functions
class DensePreproc(nn.Dense):
    min_vals: Optional[Tuple[float]] = None
    max_vals: Optional[Tuple[float]] = None

    def __call__(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        x = x.reshape((-1))  # flatten
        if self.min_vals is not None and self.max_vals is not None:
            minv = jnp.array(self.min_vals)
            x -= minv
            x /= jnp.array(self.max_vals) - minv
            x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        return super().__call__(x)


# https://flax.readthedocs.io/en/latest/design_notes/module_lifecycle.html#lifecycle
# NOTE not taking all the args to Dense
@gin.configurable
class MLP(nn.Module):
    features: int
    hiddens: Tuple[int] = (64, 64)
    activation_fn: callable = nn.relu
    min_vals: Optional[Tuple[float]] = None
    max_vals: Optional[Tuple[float]] = None

    @nn.compact
    def __call__(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        return nn.combinators.Sequential(
            [
                DensePreproc(
                    features=self.features if not self.hiddens else self.hiddens[0],
                    min_vals=self.min_vals,
                    max_vals=self.max_vals,
                )
            ]
            + [
                v
                for tup in zip(
                    [self.activation_fn] * len(self.hiddens),
                    [
                        nn.Dense(features=out_feat)
                        for out_feat in self.hiddens[1:] + (self.features,)
                    ],
                )
                for v in tup
            ]
        )(x)


nature_nn_og_call = NatureDQNNetwork.__call__


def nature_nn_new_call(self, x):
    return nature_nn_og_call(self, x).q_values


NatureDQNNetwork.__call__ = nature_nn_new_call


@gin.configurable
class EnsembledNet(nn.Module):
    model: nn.Module
    n_heads: int

    # NOTE name is required or flax's automatic naming system gets
    # tricked
    def setup(self):
        self.heads = [self.model.clone(name=f"head_{i}") for i in range(self.n_heads)]

    def __call__(
        self, x: Union[jnp.ndarray, np.ndarray], head: int = None
    ) -> jnp.ndarray:
        return (
            jnp.array([h(x) for h in self.heads])
            if head is None
            else self.heads[head](x)
        )
