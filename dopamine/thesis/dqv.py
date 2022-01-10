from dataclasses import dataclass
from typing import Dict, Tuple, Union

import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

from jax import numpy as jnp
from thesis import jax_utils as u_jax
from thesis.jax import networks


# NOTE wrap holds parameters, optimizer and optimizer state, and can be
# passed to jitted function (so it must either be a valid PyTree, or
# a node with registered type for lax to traverse)
@dataclass
class NetworkOptimWrap:
    net: nn.Module = networks.ClassicControlDNNetwork
    optim: optax.GradientTransformation = optax.sgd
    params: Union[FrozenDict, Dict[str, FrozenDict]] = None
    opt_state: optax.OptState = None


# NOTE nets and optims should take everything as kwargs!
def build_net(
    out_dim: int,
    inp_shape: Tuple[int],
    key: u_jax.PRNGKeyWrap,
    net_class: nn.Module = networks.ClassicControlDNNetwork,
    **kwargs,
) -> Tuple[nn.Module, FrozenDict]:
    net = net_class(output_dim=out_dim, **kwargs)
    params = net.init(next(key), jnp.ones(inp_shape))
    return net, params


def build_optim(
    params: FrozenDict, optim_class: optax.GradientTransformation = optax.sgd, **kwargs
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optim = optim_class(**kwargs)
    optim_state = optim.init(params)
    return optim, optim_state


@dataclass
class DQVMaxAgent:
    conf: dict
    qnet: NetworkOptimWrap = None
    vnet: NetworkOptimWrap = None

    def __post_init__(self):
        self.rng = u_jax.PRNGKeyWrap()
        self.n_actions = 2
        self.state_shape = (4,)

    def build_networks_and_optimizers(self):
        net_conf = self.conf["nets"]
        nets_out_dims = zip(["qnet", "vnet"], [self.n_actions, 1])
        for net_name, out_dim in nets_out_dims:
            model_spec = net_conf[net_name].get("model", {})
            optim_spec = net_conf[net_name].get("optim", {})
            net, params = build_net(out_dim, self.state_shape, self.rng, **model_spec)
            optim, optim_state = build_optim(params, **optim_spec)
            setattr(self, net_name, NetworkOptimWrap(net, optim, params, optim_state))
        # copy initial weights between online and target policy network
        self.qnet.params = {"online": self.qnet.params, "target": self.qnet.params}


conf = {
    "nets": {
        "qnet": {
            "model": {
                "net_class": networks.ClassicControlDNNetwork,
                "hidden_features": (2, 4),
            },
            "optim": {
                "optim_class": optax.sgd,
                "learning_rate": 0.09,
            },
        },
        "vnet": {"optim": {"learning_rate": 0.05}},
    },
    # "experiment": {},
    # "logs": {},
    # "exploration": {},
    # "memory": {},
    # "agent": {},
}

ag = DQVMaxAgent(conf)
ag.build_networks_and_optimizers()
