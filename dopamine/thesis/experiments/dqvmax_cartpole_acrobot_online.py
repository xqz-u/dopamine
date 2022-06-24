from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner


def do_confs():
    return [
        {
            "runner": runner.OnlineRunner,
            "seed": experiments.DEFAULT_SEED + i,
            "redundancy": i,
            "agent_class": agent.DQVMax,
            "env_name": env_name,
            "experiment_name": f"DQVMax_{env_name}_online_bigger_net",
            "model_maker_fn": configs.dqvmax_model_maker,
            "logs_base_dir": constants.scratch_data_dir,
            # "logs_base_dir": constants.data_dir,
            "experiment": {
                "iterations": 500,
                "steps": 1000,
                "eval_steps": 1000,
                "eval_period": 5,
            },
            "agent": {"sync_weights_every": 100},
            "model_args": {"hiddens": (256, 256)},
        }
        for env_name in ["CartPole-v1", "Acrobot-v1"]
        for i in range(1)
        # for i in range(experiments.DEFAULT_REDUNDANCY)
    ]


if __name__ == "__main__":
    confs, *_ = do_confs()
    runner.run_experiment(confs)
    # runner.run_parallel(confs)
