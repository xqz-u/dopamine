#!/usr/bin/env python3

import gin
import gym
import numpy as onp

from dopamine.jax.agents.dqn import dqn_agent

env = gym.make("CartPole-v0")

gin.parse_config_file("./dopamine/jax/agents/dqn/configs/dqn_cartpole.gin")

agent = dqn_agent.JaxDQNAgent(num_actions=env.action_space.n)

last_observation = env.reset()
action = agent.begin_episode(last_observation)

observation, reward, done, _ = env.step(action)
last_observation = onp.reshape(last_observation, agent.observation_shape)
agent._store_transition(last_observation, action, reward, False)
last_observation = observation
# agent._train_step()
action = onp.random.randint(agent.num_actions)

if done:
    agent.end_episode(reward, done)
