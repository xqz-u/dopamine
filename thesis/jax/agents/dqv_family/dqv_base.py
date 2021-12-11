#!/usr/bin/env python3

from typing import Union

import attr
import numpy as np
import optax
import tensorflow as tf
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrand

from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from thesis import experiment_data
from thesis.jax import exploration, networks
from thesis.offline.replay_memory.offline_circular_replay_buffer import (
    OfflineOutOfGraphReplayBuffer,
)


# TODO split for online and offline agents?
@attr.s(auto_attribs=True)
class DQV:
    state_shape: tuple
    num_actions: tuple
    exp_data: experiment_data.ExperimentData
    state: jnp.DeviceArray = None
    action: int = None
    memory: Union[OfflineOutOfGraphReplayBuffer, OutOfGraphReplayBuffer] = None
    training_steps = 0
    rng: jnp.DeviceArray = None
    V_network: nn.Module = networks.ClassicControlDNNetwork
    Q_network: nn.Module = networks.ClassicControlDNNetwork
    optimizer: optax.GradientTransformation = optax.sgd
    Q_optim_state: optax.OptState = None
    V_optim_state: optax.OptState = None
    summary_writer: tf.compat.v1.summary.FileWriter = None
    summary_writing_freq: int = 500

    def __attrs_post_init__(self):
        # create rng
        if self.exp_data.seed is None:
            self.exp_data.seed = int(time.time() * 1e6)
        self.rng = jrand.PRNGKey(self.exp_data.seed)
        state_shape = self.state_shape + (self.exp_data.stack_size,)
        self.state = jnp.zeros(state_shape)
        # initialize replay memory
        self.build_memory()
        # initialize neural networks and optimizer
        self.build_networks()
        self.build_optimizer()

    def build_networks(self):
        raise NotImplementedError

    def _train_step(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.optimizer = self.optimizer(self.exp_data.learning_rate)
        self.Q_optim_state = self.optimizer.init(self.Q_online)
        self.V_optim_state = self.optimizer.init(self.V_online)

    def build_memory(self):
        """Create the replay memory used by the agent. Use a
        `dopamine.replay_memory.circular_replay_buffer.OutOfGraphReplayBuffer`
        for online training mode, or a
        `thesis.offline.replay_memory.offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer`
        for offline training. The assumption is that the proper paths for
        offline buffers are given, and the buffers are loaded here.
        """
        memory_args = {
            **self.exp_data.replay_buffers_view(),
            "observation_shape": self.state_shape,
        }
        if self.exp_data.online:
            self.memory = OutOfGraphReplayBuffer(**memory_args)
            return
        self.memory = OfflineOutOfGraphReplayBuffer(**memory_args)
        self.memory.load_buffers(
            self.exp_data.checkpoint_dir, self.exp_data.checkpoint_iterations
        )

    def update_state(self, observation):
        self.state = jnp.reshape(observation, self.state_shape)

    def record_trajectory(self, reward: float, terminal: bool, episode_end=False):
        self.memory.add(
            np.asarray(self.state),
            self.action,
            reward,
            terminal,
            episode_end=episode_end,
        )

    def begin_episode(self, observation: np.ndarray) -> int:
        """
        Perform the first action. This first trajectory will be recorded in the
        next interaction step.
        """
        # initialize state with first observation
        self.update_state(observation)
        # train step
        self._train_step()
        # action selection
        self.rng, self.action = exploration.egreedy_action_selection(
            self.rng,
            self.exp_data.epsilon,
            self.num_actions,
            self.Q_network,
            self.Q_online,
            jnp.asarray(self.state),
        )
        self.action = np.asarray(self.action)
        return self.action

    # run_experiment takes care of interaction with env; here lives the rest,
    # and this routine returns the next action the agent will perform
    def step(self, reward: float, observation: np.ndarray) -> int:
        # 1. store current trajectory in replay memory
        self.record_trajectory(reward, False)
        # 2. update current state with observation
        self.update_state(observation)
        # 3. train
        self._train_step()
        # finally, choose next action and return it
        self.rng, self.action = exploration.egreedy_action_selection(
            self.rng,
            self.exp_data.epsilon,
            self.num_actions,
            self.Q_network,
            self.Q_online,
            jnp.asarray(self.state),
        )
        self.action = np.asarray(self.action)
        return self.action

    # NOTE only invoked when hitting a terminal state, not when running out of
    # time
    def end_episode(self, reward: float, terminal: bool):
        """
        Called by experiment runner when end of episode is detected.
        Simply add this last transition to the memory and signal episode end.
        """
        self.record_trajectory(reward, terminal, episode_end=True)

    def sample_memory(self, batch_size=None, indices=None) -> dict:
        return dict(
            zip(
                [
                    el.name
                    for el in self.memory.get_transition_elements(batch_size=batch_size)
                ],
                self.memory.sample_transition_batch(
                    batch_size=batch_size, indices=indices
                ),
            )
        )

    def save_summaries(self, v_loss, q_loss):
        if (
            self.summary_writer is None
            or self.training_steps % self.summary_writing_freq
        ):
            return
        self.summary_writer.add_summary(
            tf.compat.v1.Summary(
                value=[
                    tf.compat.v1.Summary.Value(tag="V-Loss", simple_value=v_loss),
                    tf.compat.v1.Summary.Value(tag="Q-Loss", simple_value=q_loss),
                ]
            ),
            self.training_steps,
        )
        self.summary_writer.flush()

    def bundle_and_checkpoint(
        self, checkpoint_dir, iteration_number, **additional_args
    ):
        if not tf.io.gfile.exists(checkpoint_dir):
            return None
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, iteration_number)
        return {
            **additional_args,
            **{
                "state": np.asarray(self.state),
                "training_steps": self.training_steps,
                "Q_optim_state": self.Q_optim_state,
                "V_optim_state": self.V_optim_state,
            },
        }
