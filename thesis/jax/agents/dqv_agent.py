import functools as ft
import time
from dataclasses import dataclass
from typing import Tuple, Union

import gin
import jax
import numpy as onp
import optax
import tensorflow as tf
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from thesis import utils as u
from thesis.experiment_data import ExperimentData
from thesis.jax import networks
from thesis.jax.agents import utils as agents_u
from thesis.offline.replay_memory.offline_circular_replay_buffer import (
    OfflineOutOfGraphReplayBuffer,
)

# NOTE many functions and class methods are called with no args when they are
# actually required; args passing is intended to be done with gin. In any case,
# ideally one would still pass the required parameters, and not pass values for
# those args that already have sensible defaults plus can be configured with gin


@ft.partial(jax.jit, static_argnums=(0, 3, 5, 7))
def train_module(
    net: nn.Module,
    params: FrozenDict,
    td_errors: jnp.DeviceArray,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: callable,
    states: onp.ndarray,
    callback: callable,
    *args,
    **kwargs
) -> [Tuple[optax.OptState, FrozenDict, float]]:
    # evaluation and loss function
    def estimate_states(params):
        estimates = jax.vmap(lambda state: net.apply(params, state))(states)
        estimates = callback(estimates.squeeze(), *args, **kwargs)
        return jnp.mean(jax.vmap(loss_fn)(td_errors, estimates))

    # optimize the network, taking the gradient of the loss function
    grad_fn = jax.value_and_grad(estimate_states)
    loss, grad = grad_fn(params)
    updates, opt_state = optim.update(grad, opt_state, loss)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


# NOTE the update in the paper by Matthia computes TD errors discriminating
# against terminal _next_states_, whereas here I am using terminal _states_
# @u.timer
@ft.partial(jax.jit, static_argnums=(0, 5))
def dqv_td_error(
    vnet: nn.Module, target_params: FrozenDict, next_states, rewards, terminals, gamma
) -> jnp.DeviceArray:
    v_values = jax.vmap(lambda state: vnet.apply(target_params, state))(next_states)
    # needed, vmap might create a column vector to vectorize operation on states
    v_values = v_values.squeeze()
    return rewards + gamma * v_values * (1 - terminals)


# @u.timer
@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def egreedy_action_selection(
    rng: jnp.DeviceArray,
    epsilon: float,
    num_actions: int,
    q_net: networks.ClassicControlDNNetwork,
    params: FrozenDict,
    state: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    key, key1, key2 = u.force_devicearray_split(rng, 3)
    return key, jnp.where(
        jrand.uniform(key1) <= epsilon,
        jrand.randint(key2, (), 0, num_actions),
        jnp.argmax(q_net.apply(params, state)),
    )


# TODO unbundle method to restart from checkpoint
# TODO eval mode
@gin.configurable
@dataclass
class JaxDQVAgent:
    state_shape: tuple
    num_actions: tuple
    exp_data: ExperimentData
    state: jnp.DeviceArray = None
    action: int = None
    memory: Union[OfflineOutOfGraphReplayBuffer, OutOfGraphReplayBuffer] = None
    # NOTE could use runner's one, but useful here for check-pointing?
    training_steps = 0
    rng: jnp.DeviceArray = None
    Q_online: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()
    V_target: FrozenDict = FrozenDict()
    V_network: nn.Module = networks.ClassicControlDNNetwork
    Q_network: nn.Module = networks.ClassicControlDNNetwork
    optimizer: optax.GradientTransformation = optax.sgd
    Q_optim_state: optax.OptState = None
    V_optim_state: optax.OptState = None
    summary_writer: tf.compat.v1.summary.FileWriter = None
    summary_writing_freq: int = 500

    def __post_init__(self):
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

    # NOTE other parameters of the networks should be already bound
    def build_networks(self):
        self.rng, rng0, rng1 = u.force_devicearray_split(self.rng, 3)
        self.V_network = self.V_network(output_dim=1)
        self.Q_network = self.Q_network(output_dim=self.num_actions)
        self.V_online = self.V_network.init(rng0, self.state)
        self.V_target = self.V_online
        self.Q_online = self.Q_network.init(rng1, self.state)

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
        memory_args = self.exp_data.replay_buffers_view() | {
            "observation_shape": self.state_shape
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
            onp.asarray(self.state),
            self.action,
            reward,
            terminal,
            episode_end=episode_end,
        )

    def begin_episode(self, observation: onp.ndarray) -> int:
        """
        Perform the first action. This first trajectory will be recorded in the
        next interaction step.
        """
        # initialize state with first observation
        self.update_state(observation)
        # train step
        self._train_step()
        # action selection
        self.rng, self.action = egreedy_action_selection(
            self.rng,
            self.exp_data.epsilon,
            self.num_actions,
            self.Q_network,
            self.Q_online,
            jnp.asarray(self.state),
        )
        self.action = onp.asarray(self.action)
        return self.action

    # run_experiment takes care of interaction with env; here lives the rest,
    # and this routine returns the next action the agent will perform
    def step(self, reward: float, observation: onp.ndarray) -> int:
        # 1. store current trajectory in replay memory
        self.record_trajectory(reward, False)
        # 2. update current state with observation
        self.update_state(observation)
        # 3. train
        self._train_step()
        # finally, choose next action and return it
        self.rng, self.action = egreedy_action_selection(
            self.rng,
            self.exp_data.epsilon,
            self.num_actions,
            self.Q_network,
            self.Q_online,
            jnp.asarray(self.state),
        )
        self.action = onp.asarray(self.action)
        return self.action

    # NOTE only invoked when hitting a terminal state, not when running out of
    # time
    def end_episode(self, reward: float, terminal: bool):
        """
        Called by experiment runner when end of episode is detected.
        Simply add this last transition to the memory and signal episode end.
        """
        self.record_trajectory(reward, terminal, episode_end=True)

    # 0. if running number of interactions with env is > than N: train phase
    # 1. sample mini-batch of size Z of transitions from replay memory
    # 2. compute TD-error
    # 3. compute loss on TD-error and train the NNs
    # 4. sync weights between online and target networks with frequency X
    def _train_step(self):
        if self.memory.add_count > self.exp_data.min_replay_history:
            # NOTE rn update every time after enough experiences have been
            # collected, the Nature DQNN paper also uses `self.exp_data.update_period`
            replay_elements = self.sample_memory()
            td_error = dqv_td_error(
                self.V_network,
                self.V_target,
                replay_elements["next_state"],
                replay_elements["reward"],
                replay_elements["terminal"],
                self.exp_data.gamma,
            )
            self.V_optim_state, self.V_online, v_loss = train_module(
                self.V_network,
                self.V_online,
                td_error,
                self.optimizer,
                self.V_optim_state,
                self.exp_data.loss_fn,
                replay_elements["state"],
                lambda estim, *_, **__: estim,
            )
            self.Q_optim_state, self.Q_online, q_loss = train_module(
                self.Q_network,
                self.Q_online,
                td_error,
                self.optimizer,
                self.Q_optim_state,
                self.exp_data.loss_fn,
                replay_elements["state"],
                lambda estim, *args, **_: agents_u.replay_chosen_q(estim, args[0]),
                replay_elements["action"],
            )
            self.save_summaries(v_loss, q_loss)
            if self.training_steps % self.exp_data.target_update_period == 0:
                self.V_target = self.V_online
        self.training_steps += 1

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

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        if not tf.io.gfile.exists(checkpoint_dir):
            return None
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, iteration_number)
        return {
            "state": onp.asarray(self.state),
            "training_steps": self.training_steps,
            "Q_online": self.Q_online,
            "V_online": self.V_online,
            "V_target": self.V_target,
            "Q_optim_state": self.Q_optim_state,
            "V_optim_state": self.V_optim_state,
        }

    @property
    def networks_shape(self):
        return dict(
            map(
                lambda attr: (
                    attr,
                    jax.tree_map(jnp.shape, getattr(self, attr)).unfreeze(),
                ),
                ["Q_online", "V_online", "V_target"],
            )
        )
