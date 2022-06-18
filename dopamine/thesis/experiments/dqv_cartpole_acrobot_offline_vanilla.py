from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner
from thesis.experiments import dqn_dqvmax_cartpole_acrobot_offline

EXPERIMENT_NAMES = ["dqv_cartpole_offline", "dqv_acrobot_offline"]

CARTPOLE_START_SEED = 12
ACROBOT_START_SEED = CARTPOLE_START_SEED
REDUNDANCY = 3

confs = [
    {
        "seed": CARTPOLE_START_SEED + i,
        "redundancy": i,
        "agent_class": agent.DQV,
        "env_name": env_name,
        "experiment_name": exp_name,
        "model_maker_fn": configs.dqvmax_model_maker,
        "logs_base_dir": constants.data_dir,
        "experiment": {
            "iterations": 1000,
            "steps": 500,
            "eval_steps": 500,
            "eval_period": 2,
        },
    }
    for env_name, exp_name in zip(["CartPole-v1", "Acrobot-v1"], EXPERIMENT_NAMES)
    for i in range(REDUNDANCY)
]


if __name__ == "__main__":
    runner.run_parallel(confs, runner.FixedBatchRunner)
