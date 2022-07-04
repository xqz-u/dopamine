import itertools as it
import os

from thesis import agent, utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, exploration, runner

exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_pres"
)
N_HEADS = 4


def dqn_model_maker_fn(env_name, n_actions, **_):
    return {
        "Q_model_def": configs.adam_mse_mlp(
            n_actions * N_HEADS,
            env_name,
            mlp={"hiddens": (512, 512)},
            info={"n_heads": N_HEADS},
        )
    }


def dqv_hydra_maker_fn(env_name, n_actions, **_):
    return {
        "V_model_def": configs.adam_mse_mlp(
            N_HEADS, env_name, mlp={"hiddens": (512, 512)}
        ),
        "Q_model_def": configs.adam_mse_mlp(
            n_actions, env_name, mlp={"hiddens": (512, 512)}
        ),
    }


def dqvmax_hydra_maker_fn(env_name, n_actions, **_):
    return {
        "V_model_def": configs.adam_mse_mlp(
            N_HEADS, env_name, mlp={"hiddens": (512, 512)}
        ),
        "Q_model_def": configs.adam_mse_mlp(
            n_actions * N_HEADS,
            env_name,
            mlp={"hiddens": (512, 512)},
            info={"n_heads": N_HEADS},
        ),
    }


agents_and_models = [
    # (agent.MultiHeadEnsembleDQN, dqn_model_maker_fn),
    # (agent.MultiHeadEnsembleDQV, dqv_hydra_maker_fn),
    (agent.MultiHeadEnsembleDQVMax, dqvmax_hydra_maker_fn)
]
redundancy = range(experiments.DEFAULT_REDUNDANCY)
redundancy_and_seeds = zip(
    redundancy, map(lambda r: r + experiments.DEFAULT_SEED, redundancy)
)
envs = ["CartPole-v1", "Acrobot-v1"]
offline_trajectories = [
    experiments.dqn_cartpole_replay_buffers_root,
    experiments.dqn_acrobot_replay_buffers_root,
]

configurables = list(
    it.product(agents_and_models, zip(envs, offline_trajectories), redundancy_and_seeds)
)


def do_confs(params: list, logs_dir):
    return [
        {
            "runner": runner.FixedBatchRunner,
            "seed": seed,
            "redundancy": repeat,
            "agent_class": ag,
            "env_name": env,
            # "experiment_name": f"fake_{exp_name_fn(ag, env)}",
            "experiment_name": exp_name_fn(ag, env),
            "model_maker_fn": model_fn,
            "logs_base_dir": logs_dir,
            "offline_root_data_dir": buffers_dir,
            "experiment": {
                "iterations": 500,
                "steps": 1000,
                "eval_steps": 1000,
                "eval_period": 5,
                # "iterations": 10,
                # "steps": 500,
                # "eval_steps": 500,
                # "eval_period": 2,
            },
            "agent": {
                "sync_weights_every": 100,
                # "sync_weights_every": 10,
                "min_replay_history": 500,
                "training_period": 4,
            },
        }
        for (ag, model_fn), (env, buffers_dir), (repeat, seed) in params
    ]


if __name__ == "__main__":
    # confs = do_confs(configurables, constants.scratch_data_dir)
    # runner.run_experiment(confs[0], 0)
    # run = runner.FixedBatchRunner(**experiments.make_conf(**confs[0]))
    # run.run()
    symp_dir = os.path.join(constants.data_dir, "symposium")
    confs = do_confs(configurables, symp_dir)
    runner.run_parallel(confs)
