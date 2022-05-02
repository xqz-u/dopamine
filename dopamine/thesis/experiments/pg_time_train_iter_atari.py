import os
import time

from thesis import config, patcher, utils
from thesis.agents import agents
from thesis.runner import runner

make_conf = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": config.classic_control_mlp_huberloss_adam,
        "vfunc": config.classic_control_mlp_huberloss_adam,
    },
    "exploration": config.egreedy_exploration,
    "agent": config.make_batch_rl_agent(agents.DQVMaxAgent),
    "memory": config.make_batch_rl_memory(parallel=False),
    "env": config.make_env("ALE/Pong", "v5", creator=patcher.create_atari_environment),
    "reporters": config.make_reporters(exp_name),
    "runner": {
        "call_": runner.FixedBatchRunner,
        "experiment": {
            "schedule": "train",
            "seed": 4,
            "steps": int(1e3),
            "iterations": 1,
        },
    },
}


def main():
    conf = make_conf("pippo")
    conf, *_ = runner.expand_conf(
        conf,
        1,
        os.path.join(config.data_dir, "Pong"),
        intermediate_dirs="replay_logs",
        iterations=[[1]],
    )
    conf["reporters"]["aim"]["repo"] = str(config.scratch_data_dir)
    utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=config.scratch_data_dir
    )
    run = runner.create_runner(conf)
    start = time.time()
    # run.run_experiment()
    run.agent.learn()
    print(f"train iteration exec time: {time.time() - start}")


def main_pg():
    conf = make_conf("pippo")
    conf, *_ = runner.expand_conf(
        conf,
        1,
        os.path.join(config.peregrine_data_dir, "Pong"),
        intermediate_dirs="replay_logs",
        iterations=[[1]],
    )
    conf["reporters"]["aim"]["repo"] = str(config.peregrine_data_dir)
    utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=config.peregrine_data_dir
    )
    run = runner.create_runner(conf)
    start = time.time()
    # run.run_experiment()
    run.agent.learn()
    print(f"train iteration exec time: {time.time() - start}")


if __name__ == "__main__":
    main_pg()
