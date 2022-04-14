import os
from typing import List

import optax
from dopamine.jax import losses
from thesis import config, exploration, networks, patcher
from thesis.agents import agents
from thesis.memory import offline_memory
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner, runner

# TODO
# max_frames_per_episode


conv_adam = {
    "model": {"call_": networks.NatureDQNNetwork},
    # "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss_metric": losses.huber_loss,
}


conf_pong_dqvmax_offline = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qfunc": conv_adam, "vfunc": conv_adam},
    "exploration": {"call_": exploration.egreedy, "epsilon_eval": 0.001},
    "agent": {
        "call_": agents.DQVMaxAgent,
        "net_sync_freq": 2000,
        # "train_freq": 4,
        # "clip_rewards": True,
        # "min_replay_history": 2000
    },
    "memory": {
        "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
        "batch_size": 256,
        # "load_parallel": False,
    },
    "env": {
        "call_": patcher.create_atari_environment,
        "environment_name": "ALE/Pong",
        "version": "v5",
        # "environment_name": "CartPole",
        # "version": "v1",
    },
    "runner": {
        "call_": FixedBatchRunner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": int(250e3),
            "eval_steps": int(125e3),
            "iterations": 200,
            "eval_period": 1,
            # "steps": 700,
            # "eval_steps": 500,
            # "iterations": 20,
            # "eval_period": 2,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 50,
            "collection_name": exp_name,
        },
        "aim": {
            "call_": reporter.AimReporter,
            "repo": str(config.data_dir),
            # "repo": str(config.scratch_data_dir),
        },
    },
}


# sample 10 replay buffers, 20% of DQN replay dataset
# (for max only like 1 or 2)
def make_configs(
    exp_name: str, repeats: int, iterations: List[List[int]]
) -> List[dict]:
    return runner.add_offline_buffers(
        runner.add_redundancies(conf_pong_dqvmax_offline(exp_name), repeats),
        os.path.join(config.data_dir, "Pong"),
        intermediate_dirs="replay_logs",
        iterations=iterations,
    )


confs = make_configs("test_pong_dqvmax_off", 5, [[1]])
# runner.p_run_experiments(confs)


# conf0, *rest_confs = expanded_confs
# from thesis import utils

# utils.data_dir_from_conf(
#     conf0["experiment_name"], conf0, basedir=config.scratch_data_dir
# )
# run = runner.create_runner(conf0)
# run.run_experiment()
