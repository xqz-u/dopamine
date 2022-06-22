import itertools as it
import os

from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner

exp_name_fn = (
    lambda *args, **kwargs: f"{experiments.base_exp_name_fn(*args, **kwargs)}_offline"
)

agents_and_models = [(agent.DQVMax, configs.dqvmax_conv_model_maker)]
envs_and_trajectories = [("ALE/Pong-v5", os.path.join(constants.data_dir, "Pong"))]
redundancy = range(experiments.DEFAULT_REDUNDANCY)
redundancy_and_seeds = zip(
    redundancy, map(lambda r: r + experiments.DEFAULT_SEED, redundancy)
)

configurables = list(
    it.product(agents_and_models, envs_and_trajectories, redundancy_and_seeds)
)


# TODO min replay history?
# TODO load only a fraction of replay data
# TODO optimal qstar_s0?
# NOTE hyperparameters taken from https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
# NOTE check with https://arxiv.org/abs/1907.04543 some hyperparameters
# - dataset size, exploration function settings
def do_confs(params: list, exp_name_fn: callable) -> list:
    return [
        {
            "runner": runner.FixedBatchRunner,
            "seed": seed,
            "redundancy": redundancy,
            "agent_class": agent_class,
            "env_name": env_name,
            "offline_root_data_dir": offline_buff_dir,
            "experiment_name": exp_name_fn(agent_class, env_name, prefix="fake_"),
            # "experiment_name": exp_name_fn(agent_class, env_name),
            "model_maker_fn": model_maker,
            "logs_base_dir": constants.scratch_data_dir,
            # "logs_base_dir": constants.data_dir,
            "experiment": {
                "iterations": 200,
                "steps": int(250e3),
                "eval_steps": int(125e3),
                "eval_period": 5,
            },
            "memory": {"batch_size": 32, "replay_capacity": int(1e6)},
            "agent": {"sync_weights_every": int(10e3)},
            "model_args": {"hiddens": (512, 512)},
        }
        for (agent_class, model_maker), (env_name, offline_buff_dir), (
            redundancy,
            seed,
        ) in params
    ]


if __name__ == "__main__":
    confs = do_confs()
    runner.run_experiment(confs)
