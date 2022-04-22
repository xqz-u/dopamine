import os

from thesis import config
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
    confs = doconfs(conf, config.data_dir)
    runner.run_experiments(confs, scratch=True)


def main_peregrine():
    runner.run_experiments(
        doconfs(make_conf("peregrine_off_time_train"), config.peregrine_data_dir)
    )


if __name__ == "__main__":
    main_peregrine()
