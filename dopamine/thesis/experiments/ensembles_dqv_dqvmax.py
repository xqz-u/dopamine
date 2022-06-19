from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner
from thesis.experiments import dqn_dqvmax_cartpole_acrobot_offline

agents_and_models = [
    (agent.DQNEnsemble, configs.dqn_ensemble_model_maker),
    (agent.DQVEnsemble, configs.dqv_ensemble_model_maker),
    (agent.DQVMaxEnsemble, configs.dqvmax_ensemble_model_maker),
]
envs_and_trajectories = [
    ("CartPole-v1", experiments.dqn_cartpole_replay_buffers_root),
    ("Acrobot-v1", experiments.dqn_acrobot_replay_buffers_root),
]

# NOTE partial the individual model_maker_fn if #heads should be
# different in some experiments, like more Q than V heads and so forth
confs = [
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
            "experiment_name": dqn_dqvmax_cartpole_acrobot_offline.exp_name_fn(
                agent_class, env_name, prefix="fake_"
            ),
            "model_maker_fn": model_maker,
            # "logs_base_dir": constants.data_dir,
            "logs_base_dir": constants.scratch_data_dir,
            "experiment": {
                "iterations": 10,
                "steps": 200,
                "eval_steps": 200,
                "eval_period": 2,
            },
            "model_args": {"heads": 2},
        }
        for i in range(experiments.DEFAULT_REDUNDANCY)
    ]
]

EXPERIMENT_NAMES = [c["experiment_name"] for c in confs]


if __name__ == "__main__":
    runner.run_parallel(confs, runner.FixedBatchRunner)
