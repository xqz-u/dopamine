import os
import time

from thesis import config, constants, utils
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
    "memory": config.make_batch_rl_memory(),
    "env": config.make_env("CartPole", "v1"),
    "reporters": config.make_reporters(exp_name, mongo_buffering=1),
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


def doconfs(conf: dict, data_basedir: str):
    return runner.expand_conf(
        conf,
        1,
        buffers_root_dir=os.path.join(
            data_basedir,
            "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
        ),
    )


def main():
    conf = make_conf("peregrine_off_time_train")
    conf, *_ = doconfs(conf, constants.data_dir)
    conf["reporters"]["aim"]["repo"] = str(constants.scratch_data_dir)
    logsdir = utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=constants.scratch_data_dir
    )
    run = runner.build_runner(conf, logsdir)
    start = time.time()
    run.run_experiment()
    print(f"train iteration exec time: {time.time() - start}")


def main_peregrine():
    conf, *_ = doconfs(
        make_conf("peregrine_off_time_train"), constants.peregrine_data_dir
    )
    conf["reporters"]["aim"]["repo"] = str(constants.peregrine_data_dir)
    utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=constants.peregrine_data_dir
    )
    run = runner.create_runner(conf)
    start = time.time()
    run.run_experiment()
    print(f"train iteration exec time: {time.time() - start}")


if __name__ == "__main__":
    main_peregrine()
    # main()
