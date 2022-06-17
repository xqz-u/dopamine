from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner

confs = [
    {
        "seed": experiments.DEFAULT_SEED + i,
        "redundancy": i,
        "agent_class": agent.DQVMax,
        "env_name": env_name,
        "experiment_name": f"DQVMax_{env_name}_online",
        "model_maker_fn": configs.dqvmax_model_maker,
        "logs_base_dir": constants.data_dir,
        "experiment": {
            "iterations": 1000,
            "steps": 500,
            "eval_steps": 500,
            "eval_period": 2,
        },
    }
    for env_name in ["CartPole-v1", "Acrobot-v1"]
    for i in range(experiments.DEFAULT_REDUNDANCY)
]


if __name__ == "__main__":
    runner.run_parallel(confs, runner.OnlineRunner)
