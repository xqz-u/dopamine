from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner


def main():
    exp_name_fn = (
        lambda ag, env, prefix="": f"{prefix}{utils.callable_name_getter(ag)}_{env}_full_experience"
    )
    agents_and_models = [(agent.DQN, configs.dqn_model_maker)]
    envs = ["CartPole-v1", "Acrobot-v1"]
    return [
        c
        for agent_class, model_maker in agents_and_models
        for env_name in envs
        for c in [
            {
                "seed": experiments.DEFAULT_SEED + i,
                "redundancy": i,
                "agent_class": agent_class,
                "env_name": env_name,
                # "experiment_name": exp_name_fn(agent_class, env_name, prefix="fake_"),
                "experiment_name": exp_name_fn(agent_class, env_name),
                "model_maker_fn": model_maker,
                "logs_base_dir": constants.data_dir,
                # "logs_base_dir": constants.scratch_data_dir,
                "experiment": {
                    "iterations": 1000,
                    "steps": 500,
                    "eval_steps": 500,
                    "eval_period": 50,
                    "record_experience": True,
                },
                # "reporters": {"mongo": {"metrics_buffering": 1}},
                # "agent": {"min_replay_history": 100},
            }
            for i in range(experiments.DEFAULT_REDUNDANCY)
        ]
    ]


# TODO net size
def confs_cp_dopamine():
    exp_name_fn = (
        lambda ag, env, prefix="": f"{prefix}{utils.callable_name_getter(ag)}_{env}_full_experience_dopamine_params"
    )
    agents_and_models = [(agent.DQN, configs.dqn_model_maker)]
    envs = ["CartPole-v1"]
    return [
        c
        for agent_class, model_maker in agents_and_models
        for env_name in envs
        for c in [
            {
                "seed": experiments.DEFAULT_SEED + i,
                "redundancy": i,
                "agent_class": agent_class,
                "env_name": env_name,
                "experiment_name": exp_name_fn(agent_class, env_name, prefix="fake_"),
                # "experiment_name": exp_name_fn(agent_class, env_name),
                "model_maker_fn": model_maker,
                # "logs_base_dir": constants.data_dir,
                "logs_base_dir": constants.scratch_data_dir,
                "experiment": {
                    "iterations": 500,
                    "steps": 1000,
                    "eval_steps": 1000,
                    "eval_period": 50,
                    "record_experience": True,
                    "schedule": "train",
                },
                "memory": {"batch_size": 128},
                "agent": {
                    "min_replay_history": 500,
                    "training_period": 4,
                    "sync_weights_every": 100,
                },
                "model_args": {"hiddens": (512, 512)},
            }
            for i in range(experiments.DEFAULT_REDUNDANCY)
        ]
    ]


confs = main()
EXPERIMENT_NAMES = [c["experiment_name"] for c in confs]

confs_dopamine = confs_cp_dopamine()


if __name__ == "__main__":
    # runner.run_parallel(confs, runner.OnlineRunner)
    runner.run_parallel(confs_dopamine, runner.OnlineRunner)
