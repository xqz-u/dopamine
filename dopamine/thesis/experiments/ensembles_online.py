import itertools as it

from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner

agents_and_models = [
    # (agent.DQNEnsemble, configs.dqn_ensemble_model_maker)
    (agent.DQVMaxEnsemble, configs.dqvmax_ensemble_model_maker),
    # (agent.DQVEnsemble, configs.dqv_ensemble_model_maker)
    # agent.DQVMaxEnsemble
]
heads = [2]
envs = ["CartPole-v1"]
configurables = list(it.product(agents_and_models, envs, heads))
redund = 1


def do_confs(redundancy=experiments.DEFAULT_REDUNDANCY):
    name_fn = lambda ag, env, prefix="": f"{prefix}{ag.__name__}_{env}_online_v0_ensure"
    return [
        c
        for (agent_class, model_maker), env_name, head in configurables
        for c in [
            {
                "runner": runner.OnlineRunner,
                "seed": experiments.DEFAULT_SEED + i,
                "redundancy": i,
                "agent_class": agent_class,
                "env_name": env_name,
                # "experiment_name": name_fn(agent_class, env_name, prefix="fake_"),
                "experiment_name": name_fn(agent_class, env_name),
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
                "agent": {
                    "sync_weights_every": 100,
                    "min_replay_history": 500,
                    "training_period": 4,
                },
                "model_args": {"hiddens": (512, 512), "heads": head},
            }
            for i in range(redundancy)
        ]
    ]


if __name__ == "__main__":
    on_confs = do_confs(redund)
    runner.run_parallel(on_confs)
