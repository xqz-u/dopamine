import gin
import gym
from dopamine.discrete_domains import gym_lib
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from thesis import experiment_data
from thesis.jax.agents.dqv_family import dqv_max_agent

env = gym.make("CartPole-v0")

gin.bind_parameter("OutOfGraphReplayBuffer.replay_capacity", 5000)
gin.bind_parameter("OutOfGraphReplayBuffer.batch_size", 128)

ag = dqn_agent.JaxDQNAgent(
    env.action_space.n,
    gin.query_parameter("gym_lib.CARTPOLE_OBSERVATION_SHAPE"),
    gin.query_parameter("jax_networks.CARTPOLE_OBSERVATION_DTYPE"),
    gin.query_parameter("gym_lib.CARTPOLE_STACK_SIZE"),
    networks.ClassicControlDQNNetwork,
)

exp_data = experiment_data.ExperimentData(
    stack_size=gin.query_parameter("gym_lib.CARTPOLE_STACK_SIZE")
)
max_ag = dqv_max_agent.JaxDQVMaxAgent(
    observation_shape=gin.query_parameter("gym_lib.CARTPOLE_OBSERVATION_SHAPE"),
    observation_dtype=env.observation_space.dtype,
    num_actions=env.action_space.n,
    exp_data=exp_data,
)

obs = env.reset()

action = ag.begin_episode(obs)
max_action = max_ag.begin_episode(obs)
