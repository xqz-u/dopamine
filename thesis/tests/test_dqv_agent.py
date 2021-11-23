from pprint import pprint

import gym

from thesis.jax.agents import dqv_agent

env = gym.make("CartPole-v0")

n_actions = env.action_space.n
state_shape = env.observation_space.shape
seed = 0

agent = dqv_agent.JaxDQVAgent(state_shape, n_actions, seed=seed)
agent.build_networks(V_features=[5, 5], Q_features=[4])

pprint(agent.networks_shape)
