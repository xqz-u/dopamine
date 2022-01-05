from typing import Sequence, Tuple, Union

import gin
from flax import linen as nn
from jax import numpy as jnp


# Basic classic control parent class network, to keep using the D*NNetworktype
# named tuple defined in atari_lib and minimize code duplication.
# NOTE this is
# very basic compared to the same one defined in dopamine.jax.networks, might
# need refinement. main difference is the use of the hidden_features
# parameters, borrowed from the flax tutorials
@gin.configurable(denylist=["output_dim"])
class ClassicControlDNNetwork(nn.Module):
    output_dim: int
    hidden_features: Sequence[int] = (512, 512)
    min_vals: Union[None, Tuple[float, ...]] = None
    max_vals: Union[None, Tuple[float, ...]] = None
    kernel_initializer: callable = nn.initializers.xavier_uniform()
    activation_fn: callable = nn.relu

    def setup(self):
        if self.min_vals is not None:
            assert self.max_vals is not None
            self._min_vals = jnp.array(self.min_vals)
            self._max_vals = jnp.array(self.max_vals)
        self.layers = [
            nn.Dense(features=feat, kernel_init=self.kernel_initializer)
            for feat in self.hidden_features
        ]
        self.final_layer = nn.Dense(
            features=self.output_dim, kernel_init=self.kernel_initializer
        )

    def __call__(self, x):
        x = x.reshape((-1))  # flatten
        if self.min_vals is not None:
            x -= self._min_vals
            x /= self._max_vals - self._min_vals
            x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        return self.final_layer(x)
