import functools as ft
import time
from typing import Dict, Tuple, Union

import attr
import jax
import numpy as np
import optax
import tensorflow as tf
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import experiment_data
from thesis import utils as u
from thesis.jax import exploration, networks
from thesis.offline.replay_memory.offline_circular_replay_buffer import (
    OfflineOutOfGraphReplayBuffer,
)

# NOTE many functions and class methods are called with no args when they are
# actually required; args passing is intended to be done with gin. In any case,
# ideally one would still pass the required parameters, and not pass values for
# those args that already have sensible defaults plus can be configured with gin


def identity(x, *_, **__):
    return x


def mask_q_estimates(q_estimates: jnp.DeviceArray, *args, **_) -> jnp.DeviceArray:
    """
    Given Q-values (a matrix of shape (replayed_states, n_actions)),
    extract the Q-values of the action chosen for replay and indexed
    by `replay_actions`.
    """

    replay_chosen_actions = args[0]
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_estimates, replay_chosen_actions)
    return replay_chosen_q


# NOTE print statements executed only once...
@ft.partial(jax.jit, static_argnums=(0, 3, 5, 7))
def train_module(
    net: nn.Module,
    params: FrozenDict,
    td_errors: jnp.DeviceArray,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: callable,
    states,
    mask: callable,
    *args,
    **kwargs,
) -> [Tuple[optax.OptState, FrozenDict, float]]:
    # evaluation and loss function
    def estimate_states(params):
        # print(params)
        # print(states)
        # print(td_errors)
        estimates = jax.vmap(lambda state: net.apply(params, state))(states)
        estimates = mask(estimates.squeeze(), *args, **kwargs)
        return jnp.mean(jax.vmap(loss_fn)(td_errors, estimates))

    # optimize the network, taking the gradient of the loss function
    grad_fn = jax.value_and_grad(estimate_states)
    loss, grad = grad_fn(params)
    updates, opt_state = optim.update(grad, opt_state, loss)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


@ft.partial(jax.jit, static_argnums=(0))
def net_eval(net: nn.Module, params: FrozenDict, inputs) -> jnp.DeviceArray:
    return jax.vmap(lambda inp: net.apply(params, inp))(inputs).squeeze()


# NOTE the update in the paper by Matthia computes TD errors discriminating
# against terminal _next_states_, whereas here I am using terminal _states_, but
# it should be the same
@ft.partial(jax.jit, static_argnums=(0, 5, 6))
def dqv_family_td_error(
    net: nn.Module,
    target_params: FrozenDict,
    next_states,
    rewards,
    terminals,
    gamma: float,
    mask: callable,
    *args,
    **kwargs,
) -> jnp.DeviceArray:
    values = jax.vmap(lambda state: net.apply(target_params, state))(next_states)
    # needed, vmap might create a column vector to vectorize operation on states
    values = values.squeeze()
    values = mask(values, *args, **kwargs)
    return rewards + gamma * values * (1 - terminals)


# TODO unbundle method to restart from checkpoint
# TODO check if an `update_period` for memory sampling gives better performance
# TODO split for online and offline agents?
@attr.s(auto_attribs=True)
class DQV:
    observation_shape: tuple
    num_actions: tuple
    exp_data: experiment_data.ExperimentData
    state: np.ndarray = None
    _observation: np.ndarray = None
    observation_dtype: np.dtype = np.dtype("float32")
    action: np.ndarray = None
    memory: Union[OfflineOutOfGraphReplayBuffer, OutOfGraphReplayBuffer] = None
    replay_elts: Dict[str, np.ndarray] = None
    training_steps: int = 0
    rng: jnp.DeviceArray = None
    V_network: nn.Module = networks.ClassicControlDNNetwork
    Q_network: nn.Module = networks.ClassicControlDNNetwork
    Q_optim_state: optax.OptState = None
    V_optim_state: optax.OptState = None
    summary_writer: tf.compat.v1.summary.FileWriter = None
    summary_writing_freq: int = 500
    eval_mode: bool = False
    _avg_loss: jnp.DeviceArray = jnp.array((0.0, 0.0))

    def __attrs_post_init__(self):
        # create rng
        if self.exp_data.seed is None:
            self.exp_data.seed = int(time.time() * 1e6)
        self.rng = jrand.PRNGKey(self.exp_data.seed)
        state_shape = self.observation_shape + (self.exp_data.stack_size,)
        # state_shape = self.observation_shape
        self.state = np.zeros(state_shape, self.observation_dtype)
        # initialize replay memory
        self.build_memory()
        # initialize neural networks and optimizer
        self.build_networks()
        if not self.eval_mode:
            self.build_optimizer()

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
            "observation_shape": self.observation_shape,
            "observation_dtype": self.observation_dtype,
        }
        if self.exp_data.online:
            self.memory = OutOfGraphReplayBuffer(**memory_args)
            return
        self.memory = OfflineOutOfGraphReplayBuffer(**memory_args)
        self.memory.load_buffers(
            self.exp_data.checkpoint_dir, self.exp_data.checkpoint_iterations
        )

    def build_networks(self) -> Tuple[jnp.DeviceArray]:
        rng, k0, k1 = u.force_devicearray_split(self.rng, 3)
        self.V_network = self.V_network(output_dim=1)
        self.Q_network = self.Q_network(output_dim=self.num_actions)
        return rng, k0, k1

    def build_optimizer(self):
        # NOTE next call needs to be bound with gin
        self.optimizer = self.exp_data.create_optimizer_fn()
        self.Q_optim_state = self.optimizer.init(self.Q_online)
        self.V_optim_state = self.optimizer.init(self.V_online)

    # the horror ....
    # 0. if running number of interactions with env is > than N: train phase
    # 1. sample mini-batch of size Z of transitions from replay memory
    # 2. specific _train_phase: compute TD-error and train NNs with its loss
    # 3. sync weights between online and target networks with frequency X
    def _train_step(self):
        if self.memory.add_count > self.exp_data.min_replay_history:
            # self.training_steps % self.exp_data.update_period == 0,
            q_loss, v_loss = self.agent_train_step(self.sample_memory())
            self._avg_loss = (self._avg_loss + jnp.array((q_loss, v_loss))) / 2
            # print(f"{self.training_steps} V loss: {v_loss} Q_loss: {q_loss}")
            if (
                self.summary_writer is not None
                and self.training_steps > 0
                and self.training_steps % self.summary_writing_freq == 0
            ):
                self.save_summaries(float(v_loss), float(q_loss))
            if self.training_steps % self.exp_data.target_update_period == 0:
                self.sync_weights()
                print("synced weights...")
        self.training_steps += 1

    def agent_train_step(self, replay_elements: dict) -> Tuple[jnp.DeviceArray]:
        raise NotImplementedError

    def sync_weights(self):
        raise NotImplementedError

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def update_state(self, observation: np.ndarray):
        self._observation = np.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self._observation

    def record_trajectory(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        episode_end=False,
    ):
        self.memory.add(
            observation,
            action,
            reward,
            terminal,
            episode_end=episode_end,
        )

    def select_action(self, obs):
        self.update_state(obs)
        self.rng, self.action = exploration.egreedy_action_selection(
            self.rng,
            self.Q_network,
            self.num_actions,
            self.eval_mode,
            self.exp_data.epsilon_train,
            self.exp_data.epsilon_eval,
            self.Q_online,
            self.state,
        )
        self.action = np.array(self.action)
        return self.action

    def learn(self, obs, reward, done):
        self.record_trajectory(self._observation, self.action, reward, done)
        if done:
            return
        self._train_step()

    def sample_memory(self, batch_size=None, indices=None):
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

    @ft.partial(jax.jit, static_argnums=(0))
    def td_error(self, estimates: jnp.DeviceArray) -> jnp.DeviceArray:
        return self.replay_elts["reward"] + self.exp_data.gamma * estimates * (
            1 - self.replay_elts["terminal"]
        )

    def save_summaries(self, v_loss: float, q_loss: float):
        u.add_aim_values(
            self.summary_writer,
            [
                [f"q-{self.exp_data.loss_fn.__name__}", q_loss],
                [f"v-{self.exp_data.loss_fn.__name__}", v_loss],
            ],
            self.training_steps,
        )
        # u.add_summary_v2(
        #     self.summary_writer,
        #     [
        #         [
        #             "scalar",
        #             f"q-{self.exp_data.loss_fn.__name__}",
        #             q_loss,
        #         ],
        #         [
        #             "scalar",
        #             f"v-{self.exp_data.loss_fn.__name__}",
        #             v_loss,
        #         ],
        #     ],
        #     self.training_steps,
        # )
        # self.summary_writer.add_summary(
        #     tf.compat.v1.Summary(
        #         value=[
        #             tf.compat.v1.Summary.Value(
        #                 tag=f"V-{self.exp_data.loss_fn.__name__}", simple_value=v_loss
        #             ),
        #             tf.compat.v1.Summary.Value(
        #                 tag=f"Q-{self.exp_data.loss_fn.__name__}", simple_value=q_loss
        #             ),
        #         ]
        #     ),
        #     self.training_steps,
        # )
        # self.summary_writer.flush()

    def bundle_and_checkpoint(
        self, checkpoint_dir, iteration_number, **additional_args
    ) -> dict:
        if not tf.io.gfile.exists(checkpoint_dir):
            return None
        # Checkpoint the out-of-graph replay buffer.
        self.memory.save(checkpoint_dir, iteration_number)
        return {
            **additional_args,
            **{
                k: getattr(self, k)
                for k in ["state", "training_steps", "Q_optim_state", "V_optim_state"]
            },
        }

        # if self.exp_data.update_period is None:
        #     self.sample_memory()
        #     q_loss, v_loss = self.agent_train_step()
        #     print(f"{self.training_steps} V loss: {v_loss} Q_loss: {q_loss}")

        # if (
        #     self.summary_writer is not None
        #     and self.training_steps > 0
        #     and self.training_steps % self.summary_writing_freq == 0
        # ):
        #     self.save_summaries(v_loss, q_loss)
        # if self.training_steps % self.exp_data.target_update_period == 0:
        #     self.sync_weights()
        #     print("synced weights...")
        # else:

        #     if self.training_steps % self.exp_data.update_period == 0:
        #         self.sample_memory()
        #         q_loss, v_loss = self.agent_train_step()
        #         print(f"{self.training_steps} V loss: {v_loss} Q_loss: {q_loss}")
        #     if (
        #         self.summary_writer is not None
        #         and self.training_steps > 0
        #         and self.training_steps % self.summary_writing_freq == 0
        #     ):
        #         self.save_summaries(v_loss, q_loss)
        #     if self.training_steps % self.exp_data.target_update_period == 0:
        #         self.sync_weights()
        #         print("synced weights...")

    # def begin_episode(self, observation: np.ndarray) -> np.ndarray:
    #     """
    #     Perform the first action. This first trajectory will be recorded in the
    #     next interaction step.
    #     """
    #     # print(f"eval_mode: {self.eval_mode}")
    #     self._reset_state()
    #     # initialize state with first observation
    #     self.update_state(observation)
    #     # print(f"state: {self.state}")
    #     # train step
    #     if not self.eval_mode:
    #         self._train_step()
    #     # action selection
    #     self.rng, self.action = exploration.egreedy_action_selection(
    #         self.rng,
    #         self.Q_network,
    #         self.num_actions,
    #         self.eval_mode,
    #         self.exp_data.epsilon_train,
    #         self.exp_data.epsilon_eval,
    #         self.Q_online,
    #         self.state,
    #     )
    #     self.action = np.asarray(self.action)
    #     return self.action

    # # run_experiment takes care of interaction with env; here lives the rest,
    # # and this routine returns the next action the agent will perform
    # def step(self, reward: float, observation: np.ndarray) -> int:
    #     # 1. update current state with observation
    #     last_observation = self._observation
    #     self.update_state(observation)
    #     if not self.eval_mode:
    #         # 2. store current trajectory in replay memory
    #         self.record_trajectory(last_observation, self.action, reward, False)
    #         # 3. train
    #         self._train_step()
    #     # finally, choose next action and return it
    #     self.rng, self.action = exploration.egreedy_action_selection(
    #         self.rng,
    #         self.Q_network,
    #         self.num_actions,
    #         self.eval_mode,
    #         self.exp_data.epsilon_train,
    #         self.exp_data.epsilon_eval,
    #         self.Q_online,
    #         self.state,
    #     )
    #     self.action = np.asarray(self.action)
    #     return self.action

    # NOTE only invoked when hitting a terminal state, not when running out of
    # time
    # def end_episode(self, reward: float, terminal: bool):
    #     """
    #     Called by experiment runner when end of episode is detected.
    #     Simply add this last transition to the memory and signal episode end.
    #     """
    #     if not self.eval_mode:
    #         self.record_trajectory(
    #             self._observation, self.action, reward, terminal, episode_end=True
    #         )
