# generate training trajectories on the MountainCar Gym environment with
# a DQN agent
from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner

START_SEED = 501

confs = [
    {
        "seed": START_SEED + i,
        "redundancy": i,
        "agent_class": agent.DQN,
        "env_name": "MountainCar-v0",
        "experiment_name": "dqn_mountaincar_record_experience",
        "model_maker_fn": configs.dqn_model_maker,
        "logs_base_dir": constants.scratch_data_dir,
        "experiment": {
            "record_experience": True,
            "iterations": 1000,
            "steps": 500,
            "eval_steps": 500,
            "eval_period": 50,
        },
    }
    for i in range(experiments.DEFAULT_REDUNDANCY)
]

if __name__ == "__main__":
    runner.run_parallel(confs, runner.OnlineRunner)
