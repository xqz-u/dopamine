import itertools as it

from thesis import agent, utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, exploration, runner

exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_pres"
)
agents_and_models = [
    (agent.BootstrappedDQN, configs.dqn_ensemble_model_maker),
    (agent.BootstrappedDQV, configs.dqv_ensemble_model_maker),
    (agent.BootstrappedDQVMax, configs.dqvmax_ensemble_model_maker),
]
heads = [5]
td_target_ensembling = [False, True]
redundancy = range(experiments.DEFAULT_REDUNDANCY)
redundancy_and_seeds = zip(
    redundancy, map(lambda r: r + experiments.DEFAULT_SEED, redundancy)
)
envs = ["CartPole-v1", "Acrobot-v1"]
offline_trajectories = [
    experiments.dqn_cartpole_replay_buffers_root,
    experiments.dqn_acrobot_replay_buffers_root,
]
envs_and_trajectories = dict(zip(envs, offline_trajectories))

configurables_online = list(
    it.product(
        agents_and_models, envs, redundancy_and_seeds, heads, td_target_ensembling
    )
)


def online_confs(params: list, logs_dir):
    return [
        {
            "runner": runner.OnlineRunner,
            "seed": seed,
            "redundancy": repeat,
            "agent_class": ag,
            "env_name": env,
            "experiment_name": f"{exp_name_fn(ag, env)}_online",
            "model_maker_fn": model_fn,
            "logs_base_dir": logs_dir,
            "experiment": {
                "iterations": 500,
                "steps": 1000,
                "eval_steps": 1000,
                "eval_period": 5,
            },
            "agent": {
                "sync_weights_every": 100,
                "min_replay_history": 500,
                "training_period": 4,
                "ensemble_td_target": ensemble_tp1,
            },
            "model_args": {"heads": n_heads, "hiddens": (512, 512)},
        }
        for (ag, model_fn), env, (repeat, seed), n_heads, ensemble_tp1 in params
    ]


def offline_confs(online_params: list, logs_dir):
    on_confs = online_confs(online_params, logs_dir)
    for c in on_confs:
        c["runner"] = runner.FixedBatchRunner
        c["offline_root_data_dir"] = envs_and_trajectories[c["env_name"]]
        c["experiment_name"] = c["experiment_name"].replace("_online", "_offline")
    return on_confs


if __name__ == "__main__":
    # exit(0)
    on_confs = online_confs(configurables_online, constants.scratch_data_dir)
    off_confs = offline_confs(configurables_online, constants.scratch_data_dir)
    all_confs = off_confs
    # all_confs = off_confs + on_confs
    runner.run_parallel(all_confs)
