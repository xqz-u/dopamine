from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

from jax import numpy as jnp
from thesis import custom_pytrees
from thesis import utils as u
from thesis.jax import exploration, networks


def build_net(
    out_dim: int,
    inp_shape: Tuple[int],
    key: custom_pytrees.PRNGKeyWrap,
    net_class: nn.Module = networks.mlp,
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
    num_actions: int = None
    state_shape: Tuple[int] = None
    action: np.ndarray = None
    state: np.ndarray = None
    qnet: custom_pytrees.NetworkOptimWrap = None
    vnet: custom_pytrees.NetworkOptimWrap = None
    training_steps: int = 0
    rng: custom_pytrees.PRNGKeyWrap = None

    def __post_init__(self):
        self.rng = custom_pytrees.PRNGKeyWrap()
        self.state_shape = (4,)
        self.num_actions = 2
        self.state = jnp.ones(self.state_shape)

    def build_networks_and_optimizers(self):
        net_conf = self.conf["nets"]
        nets_out_dims = zip(["qnet", "vnet"], [self.num_actions, 1])
        for net_name, out_dim in nets_out_dims:
            model_spec = net_conf[net_name].get("model", {})
            optim_spec = net_conf[net_name].get("optim", {})
            net, params = build_net(out_dim, self.state_shape, self.rng, **model_spec)
            optim, optim_state = build_optim(params, **optim_spec)
            setattr(
                self,
                net_name,
                custom_pytrees.NetworkOptimWrap(net, optim, params, optim_state),
            )
        # copy initial weights between online and target policy network
        # NOTE assign to vnet as well since action selection always use
        # the online parameters, but these might be in different
        # networks based on algorithm. Is this a correct design? Also
        # because I am passing a lot of args not required by an action
        # selection (e.g. target params, optim, optim state etc.),
        # although it is true that these will always be the same and
        # jit will write apprapriate code for each on the first call
        self.qnet.params = {"online": self.qnet.params, "target": self.qnet.params}
        self.vnet.params = {"online": self.vnet.params, "target": None}

    def select_action(self) -> np.ndarray:
        act_sel_fn = self.conf["exploration"]["fn"]
        act_sel_args = u.argfinder(
            act_sel_fn,
            {
                **self.conf["exploration"],
                **u.dataclass_fields_d(self),
            },
        )
        self.rng, self.action = np.array(act_sel_fn(**act_sel_args))
        self.action = np.array(self.action)
        return self.action


conf = {
    "nets": {
        "qnet": {
            "model": {
                "net_class": networks.mlp,
                "hiddens": (5, 5),
            },
            "optim": {
                "optim_class": optax.sgd,
                "learning_rate": 0.09,
            },
        },
        "vnet": {"optim": {"learning_rate": 0.05}},
    },
    "exploration": {
        # "fn": exploration.egreedy_linear_decay,
        # "decay_period": 100,
        # "warmup_steps": 800,
        # "epsilon_train": 0.02,
        "fn": exploration.egreedy,
        "eval_mode": False,
        "epsilon_train": 0.01,
        "epsilon_eval": 0.001,
    },
    # "experiment": {},
    # "logs": {},
    # "memory": {},
    # "agent": {},
}

# ag = DQVMaxAgent(conf)
# ag.build_networks_and_optimizers()
# x = ag.select_action()
