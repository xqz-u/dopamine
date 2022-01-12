from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import attr
import numpy as np
import optax
import tensorflow as tf
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, networks, utils


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


# qnet: custom_pytrees.NetworkOptimWrap = None
# vnet: custom_pytrees.NetworkOptimWrap = None
@attr.s(auto_attribs=True)
class Agent(ABC):
    conf: dict
    num_actions: int
    observation_shape: Tuple[int]
    observation_dtype: np.dtype
    action: np.ndarray = None
    state: np.ndarray = None
    models: Dict[str, custom_pytrees.NetworkOptimWrap] = {}
    memory: circular_replay_buffer.OutOfGraphReplayBuffer = None
    rng: custom_pytrees.PRNGKeyWrap = None
    training_steps: int = 0
    net_sync_freq: int = 200
    min_replay_history: int = 5000
    train_freq: int = None
    _observation: np.ndarray = None

    def __attrs_post_init__(self):
        self.rng = (
            custom_pytrees.PRNGKeyWrap()
            if not (seed := self.conf["experiment"].get("seed"))
            else custom_pytrees.PRNGKeyWrap(seed)
        )
        self.state = jnp.ones(
            self.observation_shape + (self.conf["memory"]["stack_size"],)
        )
        self.build_memory()
        self.build_networks_and_optimizers()
        # Set useful attributes such as weights syncing period etc.
        # They could still be accessed from the configuration, but this
        # way they are easier to access
        for attrib, val in self.conf["agent"].items():
            if attrib != "class_":
                setattr(self, attrib, val)

    def select_args(self, fn: callable, top_level_key: str) -> dict:
        return utils.argfinder(
            fn, {**self.conf[top_level_key], **utils.attr_fields_d(self)}
        )

    def build_memory(self):
        memory_class = self.conf["memory"].get(
            "class_", circular_replay_buffer.OutOfGraphReplayBuffer
        )
        memory_args = self.select_args(memory_class, "memory")
        self.memory = memory_class(**memory_args)

    def _build_networks_and_optimizers(
        self, net_names: Sequence[str], out_dims: Sequence[int]
    ):
        net_conf = self.conf["nets"]
        for net_name, out_dim in zip(net_names, out_dims):
            model_spec = net_conf[net_name].get("model", {})
            optim_spec = net_conf[net_name].get("optim", {})
            net, params = build_net(
                out_dim, self.observation_shape, self.rng, **model_spec
            )
            optim, optim_state = build_optim(params, **optim_spec)
            self.models[net_name] = custom_pytrees.NetworkOptimWrap(
                net, optim, params, optim_state
            )

    def record_trajectory(self, reward: float, terminal: bool):
        self.memory.add(self._observation, self.action, reward, terminal)

    def sample_memory(self) -> dict:
        return utils.sample_replay_buffer(self.memory)

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
        act_sel_fn = self.conf["exploration"]["fn"]
        act_sel_args = self.select_args(act_sel_fn, "exploration")
        self.rng, self.action = np.array(
            act_sel_fn(**act_sel_args, net=net, params=params)
        )
        self.action = np.array(self.action)
        return self.action

    def learn(self, obs: np.ndarray, reward: float, done: bool):
        self.record_trajectory(reward, done)
        if done:
            self.state.fill(0)
            return
        if self.memory.add_count > self.min_replay_history:
            if self.train_freq is None or self.training_steps % self.train_freq == 0:
                loss = self.agent_train_step(self.sample_memory())
                # self.save_summaries(loss)
            if self.training_steps % self.net_sync_freq == 0:
                self.sync_weights()
        self.training_steps += 1

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def build_networks_and_optimizers(self):
        pass

    @abstractmethod
    def agent_train_step(self):
        pass

    @abstractmethod
    def sync_weights(self):
        pass

    @abstractmethod
    def bundle_and_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> dict:
        if not tf.io.gfile.exists(checkpoint_dir):
            return
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, iteration_number)
        return {"state": self.state, "training_steps": self.training_steps}

    @abstractmethod
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        pass
        # try:
        #     # self._replay.load() will throw a NotFoundError if it does not find all
        #     # the necessary files.
        #     self._replay.load(checkpoint_dir, iteration_number)
        # except tf.errors.NotFoundError:
        #     if not self.allow_partial_reload:
        #         # If we don't allow partial reloads, we will return False.
        #         return False
        #     logging.warning("Unable to reload replay buffer!")
        # if bundle_dictionary is not None:
        #     self.state = bundle_dictionary["state"]
        #     self.training_steps = bundle_dictionary["training_steps"]
        #     if isinstance(bundle_dictionary["online_params"], core.FrozenDict):
        #         self.online_params = bundle_dictionary["online_params"]
        #         self.target_network_params = bundle_dictionary["target_params"]
        #     else:  # Load pre-linen checkpoint.
        #         self.online_params = core.FrozenDict(
        #             {
        #                 "params": checkpoints.convert_pre_linen(
        #                     bundle_dictionary["online_params"]
        #                 ).unfreeze()
        #             }
        #         )
        #         self.target_network_params = core.FrozenDict(
        #             {
        #                 "params": checkpoints.convert_pre_linen(
        #                     bundle_dictionary["target_params"]
        #                 ).unfreeze()
        #             }
        #         )
        #     # We recreate the optimizer with the new online weights.
        #     self.optimizer = create_optimizer(self._optimizer_name)
        #     if "optimizer_state" in bundle_dictionary:
        #         self.optimizer_state = bundle_dictionary["optimizer_state"]
        #     else:
        #         self.optimizer_state = self.optimizer.init(self.online_params)
        # elif not self.allow_partial_reload:
        #     return False
        # else:
        #     logging.warning("Unable to reload the agent's parameters!")
        # return True
