import itertools as it

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
exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_offline_v0"
)
# n_heads = [2, 4, 7]
n_heads = [2]
configurables = list(it.product(agents_and_models, envs_and_trajectories, n_heads))


def do_confs():
    return [
        c
        for (agent_class, model_maker), (
            env_name,
            offline_buff_dir,
        ), head in configurables
        for c in [
            {
                "runner": runner.FixedBatchRunner,
                "seed": experiments.DEFAULT_SEED + i,
                "redundancy": i,
                "agent_class": agent_class,
                "env_name": env_name,
                "offline_root_data_dir": offline_buff_dir,
                # "experiment_name": exp_name_fn(agent_class, env_name, prefix="fake_"),
                "experiment_name": exp_name_fn(agent_class, env_name),
                "model_maker_fn": model_maker,
                # "logs_base_dir": constants.scratch_data_dir,
                "logs_base_dir": constants.data_dir,
                "experiment": {
                    "iterations": 500,
                    "steps": 1000,
                    "eval_steps": 1000,
                    "eval_period": 5,
                },
                "memory": {"batch_size": 128},
                "agent": {"sync_weights_every": 100},
                "model_args": {"hiddens": (512, 512), "heads": head},
            }
            for i in range(experiments.DEFAULT_REDUNDANCY)
        ]
    ]


# initially run some online counterparts to see how they compare
def online_confs():
    confs = do_confs()
    for c in confs:
        c["runner"] = runner.OnlineRunner
        c["agent"] = {
            "min_replay_history": 500,
            "training_period": 4,
            "sync_weights_every": 100,
        }
        c.pop("offline_root_data_dir")
        c["experiment_name"] = c["experiment_name"].replace("offline", "online")
    return confs


EXPERIMENT_NAMES = lambda: [c["experiment_name"] for c in do_confs()]


if __name__ == "__main__":
    # confs = do_confs()
    # run offline and online versions at the same time
    confs = list(it.chain(*zip(do_confs(), online_confs())))
    runner.run_parallel(confs)
