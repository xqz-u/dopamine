import time
from dataclasses import dataclass
from typing import Sequence

import jax

# from dopamine.jax.agents import dqn_agent
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from thesis.jax import networks


# TODO action selection first?
# TODO sampling mechanism for replay buffer, should be able to use the one
# already implemented in the default dqn_agent
# TODO individual steps of the tranining routine
# TODO training routine, check if it is very different than the dqn_agent one, in
# any case would be cool to use the TrainingState from flax
# TODO gin config
@dataclass
class JaxDQVAgent:
    state_shape: tuple
    n_actions: tuple
    state: dict = None
    seed: int = None
    rng: jnp.DeviceArray = None
    Q_online: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()
    V_target: FrozenDict = FrozenDict()
    V_network: nn.Module = networks.ClassicControlDVNNetwork
    Q_network: nn.Module = networks.ClassicControlDQNNetwork

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(time.time() * 1e6)
        self.rng = jrand.PRNGKey(self.seed)

    def build_networks(
        self, V_features: Sequence[int] = None, Q_features: Sequence[int] = None
    ):
        state = jnp.zeros(self.state_shape)
        Vnet = (
            self.V_network(hidden_features=V_features)
            if V_features
            else self.V_network()
        )
        Qnet = (
            self.Q_network(hidden_features=Q_features, output_dim=self.n_actions)
            if Q_features
            else self.Q_network(output_dim=self.n_actions)
        )
        self.rng, _rng = jrand.split(self.rng)
        self.V_online = Vnet.init(self.rng, state)
        self.V_target = self.V_online
        self.Q_online = Qnet.init(_rng, state)
        self.V_network, self.Q_network = Vnet, Qnet
        self.state = dict(zip(["V", "Q"], [state, state]))

    @property
    def networks_shape(self):
        return {
            n: jax.tree_map(jnp.shape, p.unfreeze())
            for n, p in dict(
                map(
                    lambda attr: (attr, getattr(self, attr)),
                    ["Q_online", "V_online", "V_target"],
                )
            ).items()
        }
