from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner
from thesis.experiments import all_cartpole_acrobot_offline

agents_and_models = [
    # (agent.DQNEnsemble, configs.dqn_ensemble_model_maker),
    (agent.DQVEnsemble, configs.dqv_ensemble_model_maker),
    # (agent.DQVMaxEnsemble, configs.dqvmax_ensemble_model_maker),
]
envs_and_trajectories = all_cartpole_acrobot_offline.envs_and_trajectories
envs_and_trajectories = [envs_and_trajectories[0]]
exp_name_fn = all_cartpole_acrobot_offline.exp_name_fn


def do_confs():
    return [
        c
        for agent_class, model_maker in agents_and_models
        for env_name, offline_buff_dir in envs_and_trajectories
        for c in [
            {
                "seed": experiments.DEFAULT_SEED + i,
                "redundancy": i,
                "agent_class": agent_class,
                "env_name": env_name,
                "offline_root_data_dir": offline_buff_dir,
                "experiment_name": exp_name_fn(agent_class, env_name, prefix="fake_"),
                # "experiment_name": exp_name_fn(agent_class, env_name),
                "model_maker_fn": model_maker,
                "logs_base_dir": constants.scratch_data_dir,
                # "logs_base_dir": constants.data_dir,
                "experiment": {
                    "iterations": 500,
                    "steps": 1000,
                    "eval_steps": 1000,
                    "eval_period": 5,
                },
                "memory": {"batch_size": 128},
                "agent": {"sync_weights_every": 100},
                "model_args": {"hiddens": (512, 512), "heads": 2},
            }
            for i in range(experiments.DEFAULT_REDUNDANCY)
        ]
    ]


EXPERIMENT_NAMES = lambda: [c["experiment_name"] for c in do_confs()]

confs = do_confs()
run = runner.FixedBatchRunner(**experiments.make_conf(**confs[0]))


if __name__ == "__main__":
    confs = do_confs()
    runner.run_parallel(confs, runner.FixedBatchRunner)
