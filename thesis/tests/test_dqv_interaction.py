import os
from pprint import pp

import gin
import gym
import jax

import thesis.jax.agents.dqv_agent as dqv

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")

conf = os.path.join(THESIS, "tests", "test_dqv_interaction.gin")

gin.parse_config_file(conf)

agent = dqv.JaxDQVAgent()
dqv.build_networks(agent)
pp(agent.networks_shape)
# print(gin.operative_config_str())


def dqv_td_error(vnet, target_params, next_states, rewards, terminals, gamma):
    def td(next_state):
        return vnet.apply(target_params, next_state)

    v_values = jax.vmap(td)(next_states).v_values
    # needed, vmap creates a column vector to vectorize operation on states
    v_values = v_values.squeeze()
    return rewards + gamma * v_values * (1 - terminals)


# vnet = agent.V_network
# target_params = agent.V_target
# next_states, rewards, terminals = (
#     replay_elements["next_state"],
#     replay_elements["reward"],
#     replay_elements["terminal"],
# )

env = gym.make("CartPole-v0")

# action = agent.begin_episode(last_observation)
agent.update_state(env.reset())
# agent._train_step()
agent.rng, action = dqv.egreedy_action_selection(
    agent.rng,
    agent.exp_data.epsilon,
    agent.n_actions,
    agent.Q_network,
    agent.Q_online,
    agent.state,
)

observation, reward, done, _ = env.step(action)
if done:
    agent.end_episode(reward, done)
else:
    agent.memory.add(agent.state, action, reward, False)
    agent.update_state(observation)
    # agent._train_step()
    replay_elements = agent.sample_memory()
    err = dqv_td_error(
        agent.V_network,
        agent.V_target,
        replay_elements["next_state"],
        replay_elements["reward"],
        replay_elements["terminal"],
    )
    agent.rng, action = dqv.egreedy_action_selection(
        agent.rng,
        agent.exp_data.epsilon,
        agent.n_actions,
        agent.Q_network,
        agent.Q_online,
        agent.state,
    )
    agent.training_steps += 1
