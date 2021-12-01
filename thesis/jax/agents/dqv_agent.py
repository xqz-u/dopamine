import time
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import gin
import jax
import numpy as onp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from thesis.experiment_data import ExperimentData
from thesis.jax import networks
from thesis.offline.replay_memory.offline_circular_replay_buffer import (
    OfflineOutOfGraphReplayBuffer,
)

# NOTE many functions and class methods are called with no args when they are
# actually required; args passing is intended to be done with gin. In any case,
# ideally one would still pass the required parameters, and not pass values for
# those args that already have sensible defaults plus can be configured with gin


# onp.set_printoptions(threshold=10)


# NOTE the update in the paper by Matthia computes TD errors discriminating
# against terminal _next_states_, whereas here I am using terminal _states_
def dqv_td_error(
    vnet: nn.Module, target_params: FrozenDict, next_states, rewards, terminals, gamma
) -> jnp.DeviceArray:
    def td(next_state):
        return vnet.apply(target_params, next_state)

    v_values = jax.vmap(td)(next_states).v_values
    # needed, vmap creates a column vector to vectorize operation on states
    v_values = v_values.squeeze()
    return rewards + gamma * v_values * (1 - terminals)


# TODO mark class and `build_networks` method with gin.register; configuration
# works, but how to use it in python code? See their docs for details, using
# that decorator is also the suggested practice...
@gin.configurable
def build_networks(
    agent, V_features: Sequence[int] = None, Q_features: Sequence[int] = None
):
    state_shape = agent.state_shape + (agent.exp_data.stack_size,)
    agent.state = onp.zeros(state_shape)
    Vnet = (
        agent.V_network(hidden_features=V_features) if V_features else agent.V_network()
    )
    Qnet = (
        agent.Q_network(hidden_features=Q_features, output_dim=agent.n_actions)
        if Q_features
        else agent.Q_network(output_dim=agent.n_actions)
    )
    agent.rng, _rng = jrand.split(agent.rng)
    agent.V_online = Vnet.init(agent.rng, agent.state)
    agent.V_target = agent.V_online
    agent.Q_online = Qnet.init(_rng, agent.state)
    agent.V_network, agent.Q_network = Vnet, Qnet


# TODO @jax.jit?
def egreedy_action_selection(
    rng: jnp.DeviceArray,
    epsilon: float,
    n_actions: int,
    q_net: networks.ClassicControlDQNNetwork,
    params: FrozenDict,
    state: onp.ndarray,
) -> Tuple[jnp.DeviceArray, onp.ndarray]:
    key, key1, key2 = jrand.split(rng, 3)
    return key, onp.asarray(
        jnp.where(
            jrand.uniform(key1) <= epsilon,
            jrand.randint(key2, (), 0, n_actions),
            jnp.argmax(q_net.apply(params, state).q_values),
        )
    )


# TODO call self._train_step() iff not in eval mode
# TODO training routine, check if it is very different than the dqn_agent one, in
# any case would be cool to use the TrainingState from flax
@gin.configurable
@dataclass
class JaxDQVAgent:
    state_shape: tuple
    n_actions: tuple
    exp_data: ExperimentData
    state: onp.ndarray = None
    action: int = None
    memory: Union[OfflineOutOfGraphReplayBuffer, OutOfGraphReplayBuffer] = None
    # NOTE could use runner's one, but useful here for check-pointing?
    training_steps = 0
    rng: jnp.DeviceArray = None
    Q_online: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()
    V_target: FrozenDict = FrozenDict()
    V_network: nn.Module = networks.ClassicControlDVNNetwork
    Q_network: nn.Module = networks.ClassicControlDQNNetwork

    def __post_init__(self):
        # create rng
        if self.exp_data.seed is None:
            self.exp_data.seed = int(time.time() * 1e6)
        self.rng = jrand.PRNGKey(self.exp_data.seed)
        # initialize replay memory
        self.build_memory()
        # TODO maybe call build_networks here? if parameters are bound/passed to
        # constructor

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
        self.state = onp.reshape(observation, self.state_shape)

    # TODO
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
            self.n_actions,
            self.Q_network,
            self.Q_online,
            self.state,
        )
        return self.action

    # run_experiment takes care of interaction with env; here lives the rest,
    # and this routine returns the next action the agent will perform
    def step(self, reward: float, observation: onp.ndarray) -> int:
        # 1. store current trajectory in replay memory
        self.memory.add(self.state, self.action, reward, False)
        # 2. update current state with observation
        self.update_state(observation)
        # 3. train
        self._train_step()
        # finally, choose next action and return it
        self.rng, self.action = egreedy_action_selection(
            self.rng,
            self.exp_data.epsilon,
            self.n_actions,
            self.Q_network,
            self.Q_online,
            self.state,
        )
        return self.action

    # NOTE only invoked when hitting a terminal state, not when running out of
    # time
    def end_episode(self, reward: float, terminal: bool):
        """
        Called by experiment runner when end of episode is detected.
        Simply add this last transition to the memory and signal episode end.
        """
        self.memory.add(self.state, self.action, reward, True, episode_end=True)

    def _train_step(self):
        # 0. if running number of interactions with env is > than N: train phase
        if self.memory.add_count <= self.exp_data.min_replay_history:
            # NOTE rn update every time after enough experiences have been
            # collected, the Nature DQNN paper also uses `self.exp_data.update_period`
            return
        # 1. sample mini-batch of size Z of transitions from replay memory
        replay_elements = self.sample_memory()
        # 2. compute TD-error
        td_error = dqv_td_error(
            self.V_network,
            self.target_params,
            replay_elements["next_state"],
            replay_elements["reward"],
            replay_elements["terminal"],
            self.exp_data.gamma,
        )
        # 3. compute loss on TD-error and train the NNs
        # 4. sync weights between online and target networks with frequency X
        if not (self.training_steps % self.exp_data.target_update_period):
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
