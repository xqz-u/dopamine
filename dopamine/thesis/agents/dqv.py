from dataclasses import dataclass
from typing import Tuple

import numpy as np
import optax
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, exploration, networks
from thesis import utils as u

# TODO union between offline and online memory classes


def build_net(
    out_dim: int,
    inp_shape: Tuple[int],
    key: custom_pytrees.PRNGKeyWrap,
    class_: nn.Module = networks.mlp,
    **kwargs,
) -> Tuple[nn.Module, FrozenDict]:
    net = class_(output_dim=out_dim, **kwargs)
    params = net.init(next(key), jnp.ones(inp_shape))
    return net, params


def build_optim(
    params: FrozenDict, class_: optax.GradientTransformation = optax.sgd, **kwargs
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optim = class_(**kwargs)
    optim_state = optim.init(params)
    return optim, optim_state


@dataclass
class DQVMaxAgent:
    conf: dict
    num_actions: int = None
    observation_shape: Tuple[int] = None
    action: np.ndarray = None
    state: np.ndarray = None
    qnet: custom_pytrees.NetworkOptimWrap = None
    vnet: custom_pytrees.NetworkOptimWrap = None
    memory: circular_replay_buffer.OutOfGraphReplayBuffer = None
    rng: custom_pytrees.PRNGKeyWrap = None
    training_steps: int = 0
    net_sync_freq: int = 200
    min_replay_history: int = 5000
    train_freq: int = None

    def __post_init__(self):
        self.rng = (
            custom_pytrees.PRNGKeyWrap()
            if not (seed := self.conf["experiment"].get("seed"))
            else custom_pytrees.PRNGKeyWrap(seed)
        )
        # TODO observation_shape and num_actions are passed by the
        # runner, in config specify only environment name
        self.observation_shape = (4,)
        self.num_actions = 2
        self.state = jnp.ones(self.observation_shape)
        self.build_memory()
        self.build_networks_and_optimizers()
        # Set useful attribuets such as weights syncing period etc.
        # They could still be accessed from the configuration, but this
        # way they are easier to access
        for attr, val in self.conf["agent"].items():
            if attr != "class_":
                setattr(self, attr, val)

    def build_networks_and_optimizers(self):
        net_conf = self.conf["nets"]
        nets_out_dims = zip(["qnet", "vnet"], [self.num_actions, 1])
        for net_name, out_dim in nets_out_dims:
            model_spec = net_conf[net_name].get("model", {})
            optim_spec = net_conf[net_name].get("optim", {})
            net, params = build_net(
                out_dim, self.observation_shape, self.rng, **model_spec
            )
            optim, optim_state = build_optim(params, **optim_spec)
            setattr(
                self,
                net_name,
                custom_pytrees.NetworkOptimWrap(net, optim, params, optim_state),
            )
        # copy initial weights between online and target policy network
        self.qnet.params = {"online": self.qnet.params, "target": self.qnet.params}

    def build_memory(self):
        memory_class = self.conf["memory"].get(
            "class_", circular_replay_buffer.OutOfGraphReplayBuffer
        )
        memory_args = self.select_args(memory_class, "memory")
        self.memory = memory_class(**memory_args)

    def select_action(self) -> np.ndarray:
        # self.update_state(obs)
        act_sel_fn = self.conf["exploration"]["fn"]
        act_sel_args = self.select_args(act_sel_fn, "exploration")
        act_sel_args["params"] = self.qnet.params["online"]
        act_sel_args["net"] = self.qnet.net
        self.rng, self.action = np.array(act_sel_fn(**act_sel_args))
        self.action = np.array(self.action)
        return self.action

    def learn(self, obs: np.ndarray, reward: float, done: bool):
        self.record_trajectory(self._observation, self.action, reward, done)
        if done:
            return
        self._train_step()

    def select_args(self, fn: callable, top_level_key: str) -> dict:
        return u.argfinder(
            fn, {**self.conf[top_level_key], **u.dataclass_fields_d(self)}
        )


conf = {
    "nets": {
        "qnet": {
            "model": {
                "class_": networks.mlp,
                "hiddens": (5, 5),
            },
            "optim": {
                "class_": optax.sgd,
                "learning_rate": 0.09,
            },
        },
        "vnet": {"optim": {"learning_rate": 0.05}},
    },
    "exploration": {
        "fn": exploration.egreedy_linear_decay,
        # "fn": exploration.egreedy,
    },
    "memory": {
        "stack_size": 1,
        "replay_capacity": 50000,
        "batch_size": 128,
    },
    "experiment": {
        "seed": 4,
        "steps": 1000,
        "iterations": 500,
        "redundancy": 5,
        "env": "CartPole-v0",
    },
    "agent": {
        "class_": DQVMaxAgent,
        "net_sync_freq": 2000,
        "min_replay_history": 5000,
        "train_freq": 4,
    },
    # "logs": {},
}

ag = DQVMaxAgent(conf)
x = ag.select_action()
