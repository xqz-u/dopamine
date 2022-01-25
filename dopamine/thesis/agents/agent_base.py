from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple

import attr
import numpy as np
import tensorflow as tf
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, exploration, offline_circular_replay_buffer, utils
from thesis.agents import agent_utils


@attr.s(auto_attribs=True)
class Agent(ABC):
    conf: dict
    num_actions: int
    observation_shape: Tuple[int]
    observation_dtype: np.dtype
    net_sync_freq: int = 200
    min_replay_history: int = 5000
    train_freq: int = 1
    gamma: float = 0.99
    memory: circular_replay_buffer.OutOfGraphReplayBuffer = (
        circular_replay_buffer.OutOfGraphReplayBuffer
    )
    act_sel_fn: callable = exploration.egreedy
    eval_mode: bool = False
    model_names: Tuple[str] = attr.ib(factory=list)
    models: Dict[str, custom_pytrees.NetworkOptimWrap] = attr.ib(factory=dict)
    action: np.ndarray = None
    state: np.ndarray = None
    rng: custom_pytrees.PRNGKeyWrap = None
    training_steps: int = 0
    _observation: np.ndarray = None

    # TODO hash class and cache this
    @property
    def losses_names(self) -> Tuple[str]:
        return tuple(
            f"{m}_{self.models[m].loss_metric.__name__}" for m in self.model_names
        )

    @property
    def rl_mode(self) -> bool:
        return (
            "offline"
            if isinstance(
                self.memory,
                offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
            )
            else "online"
        )

    def __attrs_post_init__(self):
        self.rng = self.rng or custom_pytrees.PRNGKeyWrap()
        self.conf["memory"]["stack_size"] = self.conf["memory"].get("stack_size", 1)
        self.state = jnp.ones(
            self.observation_shape + (self.conf["memory"]["stack_size"],)
        )
        self.build_memory()
        self.build_networks_and_optimizers()
        self.act_sel_fn = self.conf["exploration"].get("call_", self.act_sel_fn)
        self.conf["exploration"] = {
            "call_": self.act_sel_fn,
            **utils.callable_defaults(self.act_sel_fn),
        }
        self.conf["exploration"].pop("eval_mode")
        self.conf["agent"].update(
            {
                k: getattr(self, k)
                for k in ["net_sync_freq", "min_replay_history", "train_freq", "gamma"]
            }
        )

    def select_args(self, fn: callable, top_level_key: str) -> dict:
        return utils.argfinder(
            fn, {**self.conf[top_level_key], **utils.attr_fields_d(self)}
        )

    def build_memory(self):
        memory_class = self.conf["memory"].get("call_", self.memory)
        args = self.select_args(memory_class, "memory")
        self.conf["memory"] = args
        self.memory = memory_class(**args)
        if memory_class is offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer:
            self.memory.load_buffers()

    # NOTE should the static args to loss_metric be partialled?
    # consider that it can be done before passing the function in the
    # config at the very start...
    def _build_networks_and_optimizers(
        self, net_names: Sequence[str], out_dims: Sequence[int]
    ):
        net_conf = self.conf["nets"]
        for net_name, out_dim in zip(net_names, out_dims):
            model_spec = net_conf[net_name].get("model", {})
            optim_spec = net_conf[net_name].get("optim", {})
            net, params, net_conf[net_name]["model"] = agent_utils.build_net(
                out_dim, self.observation_shape, self.rng, **model_spec
            )
            optim, optim_state, net_conf[net_name]["optim"] = agent_utils.build_optim(
                params, **optim_spec
            )
            loss_fn = net_conf[net_name].get(
                "loss", custom_pytrees.NetworkOptimWrap.loss
            )
            net_conf[net_name]["loss"] = loss_fn
            self.models[net_name] = custom_pytrees.NetworkOptimWrap(
                params, optim_state, net, optim, loss_fn
            )

    def record_trajectory(self, reward: float, terminal: bool):
        self.memory.add(self._observation, self.action, reward, terminal)

    def sample_memory(self) -> dict:
        return agent_utils.sample_replay_buffer(self.memory)

    # taken from dopamine.jax.agents.dqn.dqn_agent
    def update_state(self, obs: np.ndarray):
        # Set current observation. We do the reshaping to handle
        # environments without frame stacking.
        self._observation = np.reshape(obs, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self._observation

    def _select_action(
        self, obs: np.ndarray, net: nn.Module, params: FrozenDict
    ) -> np.ndarray:
        self.update_state(obs)
        self.rng, self.action = np.array(
            self.act_sel_fn(
                **self.select_args(self.act_sel_fn, "exploration"),
                net=net,
                params=params,
            )
        )
        self.action = np.array(self.action)
        return self.action

    def learn(
        self, obs: np.ndarray, reward: float, done: bool
    ) -> Optional[jnp.DeviceArray]:
        if done:
            self.state.fill(0)
        if self.eval_mode:
            return
        return getattr(self, f"learn_{self.rl_mode}")(obs, reward, done)

    def learn_online(
        self, obs: np.ndarray, reward: float, done: bool
    ) -> Optional[jnp.DeviceArray]:
        self.record_trajectory(reward, done)
        return self.fit(obs, reward, done)

    def fit(
        self, obs: np.ndarray, reward: float, done: bool
    ) -> Optional[jnp.DeviceArray]:
        losses = None
        if self.memory.add_count > self.min_replay_history:
            if self.training_steps % self.train_freq == 0:
                losses = self.train(self.sample_memory()).reshape((len(self.models), 1))
            if self.training_steps % self.net_sync_freq == 0:
                self.sync_weights()
        self.training_steps += 1
        return losses

    def bundle_and_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> dict:
        if not tf.io.gfile.exists(checkpoint_dir):
            return
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, iteration_number)
        return {
            "state": self.state,
            "training_steps": self.training_steps,
            "models": {
                model: self.models[model].checkpointable_elements
                for model in self.model_names
            },
        }

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary) -> bool:
        try:
            self.memory.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            pass
            # logging.warning("Unable to reload replay buffer!")
        if bundle_dictionary is None:
            # logging.warning("Unable to reload the agent's parameters!")
            return False
        self.state = bundle_dictionary["state"]
        self.training_steps = bundle_dictionary["training_steps"]
        for model_name, model in bundle_dictionary["models"].items():
            for field_name, val in model.items():
                assert model_name in self.model_names
                setattr(self.models[model_name], field_name, val)
        return True

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
        pass

    @abstractmethod
    def build_networks_and_optimizers(self):
        pass

    @abstractmethod
    def sync_weights(self):
        pass
