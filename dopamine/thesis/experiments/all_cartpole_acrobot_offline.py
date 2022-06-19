from thesis import utils

utils.setup_root_logging()


from thesis import agent, configs, constants, experiments, runner

exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_offline_v1"
)
agents_and_models = [
    (agent.DQN, configs.dqn_model_maker),
    (agent.DQVMax, configs.dqvmax_model_maker),
    (agent.DQV, configs.dqvmax_model_maker),
]
envs_and_trajectories = [
    ("CartPole-v1", experiments.dqn_cartpole_replay_buffers_root),
    ("Acrobot-v1", experiments.dqn_acrobot_replay_buffers_root),
]


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
                "model_args": {"hiddens": (512, 512)},
            }
            for i in range(experiments.DEFAULT_REDUNDANCY)
        ]
    ]


EXPERIMENT_NAMES = lambda: [c["experiment_name"] for c in do_confs()]


if __name__ == "__main__":
    confs = do_confs()
    runner.run_parallel(confs, runner.FixedBatchRunner)
