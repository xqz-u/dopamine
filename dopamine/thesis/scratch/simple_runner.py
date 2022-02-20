import os
import time

from thesis import utils as u


def run_iter(agent, env, steps, loss_dump_freq, aim_logger):
    run_steps, n_episodes, run_rwd = 0, 0, 0.0
    while run_steps < steps:
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            losses = agent.learn(obs, reward, done)
            if agent.training_steps % loss_dump_freq == 0 and losses is not None:
                u.add_aim_values(aim_logger, losses, agent.training_steps)
            run_steps += 1
            run_rwd += reward
        n_episodes += 1
    return run_rwd, run_steps, n_episodes


def run_exp(iters, steps, agent, env, loss_dump_freq, aim_logger):
    for i in range(iters):
        start = time.time()
        run_rwd, run_steps, n_episodes = run_iter(
            agent, env, steps, loss_dump_freq, aim_logger
        )
        mean_steps_per_second = run_steps / (time.time() - start)
        mean_rwd = run_rwd / n_episodes
        u.add_aim_values(
            aim_logger,
            [
                ["Train/AverageReturns", mean_rwd],
                ["Train/AverageStepsPerSecond", mean_steps_per_second],
                ["Train/NumEpisodes", n_episodes],
            ],
            i,
        )
        # utils.add_summary_v2(
        #     agent.summary_writer,
        #     [
        #         ["scalar", "Train/AverageReturns", mean_rwd],
        #         ["scalar", "Train/AverageStepsPerSecond", mean_steps_per_second],
        #         ["scalar", "Train/NumEpisodes", n_episodes],
        #     ],
        #     i,
        #     flush=True,
        # )
        print(f"{i}-{run_steps}: rwd {mean_rwd} mean_loss {agent._avg_loss}")


def make_unique_data_dir(experiment_spec: list, base_dir: str = None) -> str:
    agent, net, env_name, *args = experiment_spec
    return os.path.join(
        base_dir,
        f"{agent}_{net}_{env_name}_{'_'.join(map(str, args))}_{int(time.time())}",
    )
