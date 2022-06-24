import itertools as it

from thesis import agent, utils

# import logging


# utils.setup_root_logging(logging.DEBUG)
utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, exploration, runner

exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_debug_online_v2"
)

# agents_and_models = [(agent.DQNEnsemble, configs.dqn_ensemble_model_maker)]
agents_and_models = [(agent.BootstrappedDQN, configs.dqn_ensemble_model_maker)]
envs = ["CartPole-v1"]
heads = [10]
# heads = [2]
redundancy = range(1)
redundancy_and_seeds = zip(
    redundancy, map(lambda r: r + experiments.DEFAULT_SEED, redundancy)
)

configurables = list(it.product(agents_and_models, envs, redundancy_and_seeds, heads))


def do_confs(params: list):
    return [
        {
            "runner": runner.OnlineRunner,
            "seed": seed,
            "redundancy": 1,
            "agent_class": ag,
            "env_name": env,
            "experiment_name": exp_name_fn(ag, env, prefix="fake_"),
            # "experiment_name": exp_name_fn(ag, env),
            "model_maker_fn": model_fn,
            "logs_base_dir": constants.scratch_data_dir,
            # "logs_base_dir": constants.data_dir,
            "experiment": {
                "iterations": 1000,
                "steps": 500,
                "eval_steps": 500,
                "eval_period": 5,
            },
            "memory": {"batch_size": 32},
            "agent": {
                "sync_weights_every": 100,
                "min_replay_history": 500,
                "training_period": 1,
                "ensemble_td_target": True,
            },
            # "model_args": {"heads": n_heads, "hiddens": (512, 512)},
            "model_args": {"heads": n_heads, "hiddens": (10, 10)},
        }
        for (ag, model_fn), env, (repeat, seed), n_heads in params
    ]


conf, *_ = do_confs(configurables)
# runner.run_experiment(conf)
runner_class = conf.pop("runner")
run = runner_class(**experiments.make_conf(**conf))
run.run()
# runner.run_experiment(conf)
