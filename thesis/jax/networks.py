from collections import namedtuple
from typing import Sequence

import gin
from flax import linen as nn

from dopamine.discrete_domains import atari_lib


# Basic classic control parent class network, to keep using the D*NNetworktype
# named tupel defined in atari_lib and minimize code duplication.
# NOTE this is
# very basic compared to the same one defined in dopamine.jax.networks, might
# need refinement. main difference is the use of the hidden_features
# parameters, borrowed from the flax tutorials
# TODO test gin configurability
class ClassicControlDNNetwork(nn.Module):
    output_wrap: namedtuple
    output_dim: int = 1
    hidden_features: Sequence[int] = (512,)
    kernel_initializer: callable = nn.initializers.xavier_uniform()
    activation_fn: callable = nn.relu

    def setup(self):
        self.layers = [
            nn.Dense(features=feat, kernel_init=self.kernel_initializer)
            for feat in self.hidden_features
        ]
        self.final_layer = nn.Dense(
            features=self.output_dim, kernel_init=self.kernel_initializer
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        return self.output_wrap(self.final_layer(x))


@gin.configurable
class ClassicControlDQNNetwork(ClassicControlDNNetwork):
    """Jax DQN network for classic control environments."""

    output_wrap: namedtuple = atari_lib.DQNNetworkType


@gin.configurable(denylist=["output_dim"])
class ClassicControlDVNNetwork(ClassicControlDNNetwork):
    """Jax DVN network for classic control environments."""

    output_wrap: namedtuple = atari_lib.DVNNetworkType
