import os
import time

from thesis import utils

# def run(agent, env):
#     for i in range(500):
#         run_r = 0.0
#         obs = env.reset()
#         done = False
#         while not done:
#             action = agent.select_action(obs)
#             obs, reward, done, _ = env.step(action)
#             run_r += reward
#             agent.learn(obs, reward, done)
#         print(run_r)
#         if i % 10 == 0:
#             agent.q_net_target.load_state_dict(agent.q_net_online.state_dict())


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


def make_unique_data_dir(experiment_spec: list, base_dir: str = None) -> str:
    agent, net, env_name, *args = experiment_spec
    return os.path.join(
        base_dir,
        f"{agent}_{net}_{env_name}_{'_'.join(map(str, args))}_{int(time.time())}",
    )
