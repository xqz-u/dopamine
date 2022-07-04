import operator
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Tuple, Union

import jax
import numpy as np
from attrs import define, field
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import custom_pytrees, exploration, memory, types
from thesis.agent import utils as agent_utils


# NOTE some attrs.field are given a default in order to exist in an
# instance, or they wouldn't until explicitly set. Conversely, those
# fields with only init=False are created in __attrs_post_init__
@define
class Agent(ABC):
    policy_evaluator: exploration.PolicyEvaluator
    memory: memory.OutOfGraphReplayBuffer
    rng: custom_pytrees.PRNGKeyWrap
    clip_rewards: bool = True
    gamma: float = 0.99
    min_replay_history: int = int(2e4)
    sync_weights_every: int = int(5e3)
    training_period: int = 1
    action: np.ndarray = field(init=False, default=None)
    curr_observation: np.ndarray = field(init=False, default=None)
    # NOTE PyTreeNode(s) saved in 'models' are usually instances of (or
    # contain instances of) flax.training.train_state.TrainState; create
    # viable typesig if the available TS types change
    models: Dict[
        str, Union[custom_pytrees.ValueBasedTS, Iterable[custom_pytrees.ValueBasedTS]]
    ] = field(init=False, factory=dict)
    train_fn: Callable[
        [float, custom_pytrees.ValueBasedTS, Dict[str, np.ndarray]],
        Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS],
    ] = field(init=False)
    training_steps: int = field(init=False, default=0)
    state: np.ndarray = field(init=False)
    name: str = field(init=False)

    def __attrs_post_init__(self):
        # a state is a stack of frames in e.g. the atari environments
        self.state = jnp.zeros(
            self.observation_shape + (self.memory._stack_size,),
            dtype=self.memory._observation_dtype,
        )
        self.name = type(self).__name__

    @property
    @abstractmethod
    def act_selection_params(self) -> frozen_dict.FrozenDict:
        """Model weights used by the policy network."""
        ...

    @property
    @abstractmethod
    def initial_train_dict(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        """
        A dictionary with the same structure returned by self.train(),
        used when no training occurs.
        """
        ...

    @abstractmethod
    def train(
        self, experience_batch: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        """
        Classe which extend Agent should perform arbitrary train passes
        in this method, which receives a batch of replay trajectories;
        the learner's model(s) should be updated here, and the return
        dictionary must have the same structure as that returned by
        self.initial_train_dict. During iterative training, the returned
        dictionary is incrementally updated by self.train_accumulate
        """
        ...

    @abstractmethod
    def sync_weights(self):
        """
        Method used to arbitrarily update a target model with an online
        one, if any.
        """
        ...

    def on_episode_start(self, mode: str):
        """Simple event that can be overridden for each agent."""
        ...

    # NOTE default implementation to be overridden to match the
    # structure of the dictionaries at self.initial_train_dict and
    # self.train
    def train_accumulate(
        self, old: types.MetricsDict, new: types.MetricsDict
    ) -> types.MetricsDict:
        old["loss"] = jax.tree_map(operator.add, old["loss"], new["loss"])
        return old

    # NOTE train_dict could also be passed to self.train to modify it
    def learn(self) -> Dict[str, jnp.ndarray]:
        train_dict = self.initial_train_dict
        if self.memory.add_count > self.min_replay_history:
            if not self.training_steps % self.training_period:
                train_dict = self.train(self.sample_memory())
            if not self.training_steps % self.sync_weights_every:
                self.sync_weights()
        self.training_steps += 1
        return train_dict

    # # choose the next action; mode is "train/eval"
    def select_action(
        self, obs: np.ndarray, mode: str
    ) -> Tuple[np.ndarray, types.MetricsDict]:
        self.update_state(obs)
        self.rng, action, additional_info = self.policy_evaluator(
            self.rng,
            mode,
            self.act_selection_params,
            self.state,
            training_steps=self.training_steps,
        )
        self.action = np.array(action)
        return self.action, additional_info

    def record_trajectory(self, reward: float, terminal: bool):
        if self.clip_rewards:
            reward = np.clip(reward, -1, 1)
        self.memory.add(
            self.curr_observation,
            self.action,
            reward,
            terminal,
        )

    # taken from dopamine.jax.agents.dqn.dqn_agent
    def update_state(self, obs: np.ndarray):
        # Set current observation. We do the reshaping to handle
        # environments without frame stacking.
        self.curr_observation = np.reshape(obs, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self.curr_observation

    def sample_memory(self) -> Dict[str, np.ndarray]:
        return agent_utils.sample_replay_buffer(self.memory)

    @property
    def serializable(self) -> dict:
        return {
            "training_steps": self.training_steps,
            "rng": self.rng.serializable,
            "models": {
                model_name: model_TS.serializable
                for model_name, model_TS in self.models.items()
            },
        }

    @property
    def observation_shape(self) -> Tuple[int]:
        return self.memory._observation_shape

    @property
    def reportable(self) -> Tuple[str]:
        return (
            "policy_evaluator",
            "rng",
            "memory",
            "clip_rewards",
            "gamma",
            "min_replay_history",
            "sync_weights_every",
            "training_period",
        )
