import itertools as it
import os

from thesis import config, constants
from thesis.agents import agents
from thesis.runner import runner

# really TODO: argparser...
# NOTE i dont have mountaincar offline data yet!!!
games = [
    ("CartPole-v1", "cp"),
    ("Acrobot-v1", "ab"),
    # "MountainCar-v0"
]

experiments = {
    (agents.DQVAgent, "dqv"): games,
    (agents.DQNAgent, "dqn"): games,
    # agents.DQVMaxAgent: [games[-1]],
}


make_conf = lambda exp_name, env_version, agent_class, use_vfunc: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": config.classic_control_mlp_huberloss_adam,
        **(
            {}
            if not use_vfunc
            else {"vfunc": config.classic_control_mlp_huberloss_adam}
        ),
    },
    "exploration": config.egreedy_exploration,
    "agent": config.make_batch_rl_agent(agent_class),
    "memory": config.make_batch_rl_memory(),
    "env": config.make_env(*env_version.split("-")),
    "reporters": config.make_reporters(exp_name),
    "runner": {
        "call_": runner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": int(1e3),
            "iterations": 500,
            "eval_period": 2,
        },
    },
}


def classic_confs(logsdir, off_data_dir) -> list:
    confs = []
    for key in experiments:
        for gym_game, game_prefix in experiments[key]:
            agent_class, agent_prefix = key
            confs.append(
                runner.expand_conf(
                    make_conf(
                        f"{game_prefix}_{agent_prefix}_distr_shift_baseline",
                        gym_game,
                        agent_class,
                        agent_class != agents.DQNAgent,
                    ),
                    3,
                    buffers_root_dir=os.path.join(
                        off_data_dir,
                        f"{gym_game}/DQNAgent/{game_prefix}_dqn_full_experience_%%/checkpoints/full_experience",
                    ),
                )
            )
    return list(it.chain(*confs))


if __name__ == "__main__":
    # logs_dir, off_data_dir = constants.scratch_data_dir, constants.data_dir
    logs_dir, off_data_dir = [constants.peregrine_data_dir] * 2
    confs = classic_confs(logs_dir, off_data_dir)
    runner.p_run_experiments(confs, logs_dir=logs_dir)
