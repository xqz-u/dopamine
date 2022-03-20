from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import attr
import numpy as np
import tensorflow as tf
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import (
    custom_pytrees,
    exploration,
    offline_circular_replay_buffer,
    patcher,
    utils,
)
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
    memory: patcher.OutOfGraphReplayBuffer = patcher.OutOfGraphReplayBuffer
    act_sel_fn: callable = exploration.egreedy
    eval_mode: bool = False
    models: Dict[str, custom_pytrees.NetworkOptimWrap] = attr.ib(factory=dict)
    action: np.ndarray = None
    state: np.ndarray = None
    rng: custom_pytrees.PRNGKeyWrap = None
    training_steps: int = 0
    loss_names: Tuple[str] = None
    _observation: np.ndarray = None

    @property
    def trainable(self) -> bool:
        return self.memory.add_count > self.min_replay_history

    def __attrs_post_init__(self):
        self.rng = self.rng or custom_pytrees.PRNGKeyWrap()
        self.conf["memory"]["stack_size"] = self.conf["memory"].get("stack_size", 1)
        self.state = jnp.ones(
            self.observation_shape + (self.conf["memory"]["stack_size"],)
        )
        self.build_memory()
        self.build_networks_and_optimizers()
        self.loss_names = tuple(
            f"{m}_{self.models[m].loss_metric.__name__}" for m in self.model_names
        )
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
        memory_class = self.conf["memory"].pop("call_", self.memory)
        args = self.select_args(memory_class, "memory")
        self.conf["memory"] = args
        self.memory = memory_class(**args)
        self.conf["memory"]["call_"] = memory_class
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
            model_spec = {
                **net_conf[net_name].get("model", {}),
                **self.conf["env"].get("preproc", {}),
            }
            optim_spec = net_conf[net_name].get("optim", {})
            net, params, net_conf[net_name]["model"] = agent_utils.build_net(
                out_dim, self.state, self.rng, **model_spec
            )
            optim, optim_state, net_conf[net_name]["optim"] = agent_utils.build_optim(
                params, **optim_spec
            )
            loss_fn = net_conf[net_name].get(
                "loss_metric", custom_pytrees.NetworkOptimWrap.loss_metric
            )
            net_conf[net_name]["loss_metric"] = loss_fn
            self.models[net_name] = custom_pytrees.NetworkOptimWrap(
                params, optim_state, net, optim, loss_fn
            )

    def init_loss(self) -> jnp.DeviceArray:
        return jnp.zeros((len(self.conf["nets"]), 1))

    def record_trajectory(self, reward: float, terminal: bool):
        if not self.eval_mode:
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
        self.rng, self.action = self.act_sel_fn(
            **self.select_args(self.act_sel_fn, "exploration"),
            net=net,
            params=params,
        )
        self.action = np.array(self.action)
        return self.action

    def learn(self) -> Dict[str, jnp.DeviceArray]:
        train_dict = {"loss": self.init_loss(), "q_estimates": jnp.array(0)}
        if self.trainable:
            if self.training_steps % self.train_freq == 0:
                train_dict = self.train(self.sample_memory())
                train_dict["loss"] = jnp.array(train_dict["loss"]).reshape(
                    len(self.models), 1
                )
            if self.training_steps % self.net_sync_freq == 0:
                self.sync_weights()
        self.training_steps += 1
        return train_dict

    def bundle_and_checkpoint(self, ckpt_dir: str, iteration: int) -> dict:
        if not tf.io.gfile.exists(ckpt_dir):
            return
        # # Checkpoint the replay buffer.
        self.memory.save(ckpt_dir, iteration)
        # NOTE checkpointing happens after a full iteration, when state
        # is reset to 0s, so no use in saving it
        return {
            "training_steps": self.training_steps,
            "rng": self.rng.checkpointable_elements,
            "models": {
                model: self.models[model].checkpointable_elements
                for model in self.model_names
            },
        }

    # TODO transform into a classmethod to avoid running initializer
    # code when restoring from a checkpoint, since attributes created
    # with build_memory and stuff are overwritten here
    def unbundle(self, ckpt_dir: str, iteration: int, bundle_dict: dict):
        try:
            self.memory.load(ckpt_dir, iteration)
        except tf.errors.NotFoundError:
            # logging.warning("Unable to reload replay buffer!")
            pass
        self.training_steps = bundle_dict["training_steps"]
        for model_name, model in bundle_dict["models"].items():
            for field_name, val in model.items():
                assert model_name in self.model_names
                setattr(self.models[model_name], field_name, val)

    @property
    @abstractmethod
    def model_names(self) -> Sequence[str]:
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        pass

    # NOTE the order that losses are returned must match that in which
    # model_names are declared, since they will be zipped together
    @abstractmethod
    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        pass

    @abstractmethod
    def build_networks_and_optimizers(self):
        pass

    @abstractmethod
    def sync_weights(self):
        pass
