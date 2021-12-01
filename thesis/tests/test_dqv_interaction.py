import os

import gin
import gym

import thesis.jax.agents.dqv_agent as dqv

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")

conf = os.path.join(THESIS, "tests", "test_dqv_interaction.gin")

gin.parse_config_file(conf)


agent = dqv.JaxDQVAgent()
dqv.build_networks(agent)
dqv.build_optimizer(agent)

# pp(agent.networks_shape)
# print(gin.operative_config_str())
env = gym.make("CartPole-v0")
action = agent.begin_episode(env.reset())

partial_steps, total_reward = 1, 0
steps = 10000
for _ in range(steps):
    observation, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        print("------------------------------------------- done")
        print(f"steps: {partial_steps}, reward: {total_reward}")
        print("------------------------------------------------")
        agent.end_episode(reward, done)
        action = agent.begin_episode(env.reset())
        partial_steps, total_reward = 1, 0
    else:
        action = agent.step(reward, observation)
        partial_steps += 1
