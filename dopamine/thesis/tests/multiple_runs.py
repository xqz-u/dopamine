import time

import gin
import gym
import optax
import tensorflow as tf
from dopamine.discrete_domains import run_experiment
from dopamine.replay_memory import circular_replay_buffer
from thesis import utils
from thesis.tests import dqn_jax


def run_iter(agent, env, steps):
    run_steps, n_episodes, run_rwd = 0, 0, 0.0
    while run_steps < steps:
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.learn(obs, reward, done)
            run_steps += 1
            run_rwd += reward
        n_episodes += 1
    return run_rwd, run_steps, n_episodes


def run_exp(iters, steps, agent, env):
    for i in range(iters):
        start = time.time()
        run_rwd, run_steps, n_episodes = run_iter(agent, env, steps)
        mean_steps_per_second = run_steps / (time.time() - start)
        mean_rwd = run_rwd / n_episodes
        utils.add_summary_v2(
            agent.summary_writer,
            [
                ["scalar", "Train/AverageReturns", mean_rwd],
                ["scalar", "Train/AverageStepsPerSecond", mean_steps_per_second],
                ["scalar", "Train/NumEpisodes", n_episodes],
            ],
            i,
            flush=True,
        )
        print(f"{i}-{run_steps}: rwd {mean_rwd} mean_loss {agent._avg_loss}")


# egg
def repeat_ref(n, fold):
    gin.parse_config_file(
        "/home/xqz-u/uni/thesis/dopamine/dopamine/jax/agents/dqn/configs/dqn_cartpole.gin"
    )
    for i in range(1, n):
        runner = run_experiment.create_runner(f"{fold}/ref_JaxDQN_{i}")
        # runner._agent.seed = 0
        runner._agent.seed = i
        runner.run_experiment()
        print(f"REF {i} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def repeat_exp(n, fold):
    for i in range(n):
        env = gym.make("CartPole-v0")
        obs_shape, action_n = env.observation_space.shape, env.action_space.n
        jax_agent = dqn_jax.DQNAgent(
            action_n,
            obs_shape,
            seed=0,
            memory=circular_replay_buffer.OutOfGraphReplayBuffer(
                obs_shape, 1, 50000, 128, observation_dtype=env.observation_space.dtype
            ),
            q_net=dqn_jax.mlp([512, 512, action_n]),
            optim=optax.adam(learning_rate=0.001, eps=3.125e-4),
            summary_writer=tf.summary.create_file_writer(f"{fold}/JaxDQN_{i}"),
        )
        run_exp(500, 1000, jax_agent, env)
        print(f"MINE {i} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


data_dir = "/home/xqz-u/uni/thesis/dopamine/thesis/mul_runs"

# repeat_exp(10, data_dir)
repeat_ref(10, data_dir)
