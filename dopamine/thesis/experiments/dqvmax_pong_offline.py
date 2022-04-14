import logging
import os

import optax
from dopamine.jax import losses
from thesis import config, exploration, patcher
from thesis.agents import DQVMaxAgent
from thesis.memory import offline_memory
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner

# TODO
# death by max_episode_steps?
# test online to see that exploration works and clip_rewards too


conv_adam = {
    "model": {"call_": patcher.NatureDQNNetwork},
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss_metric": losses.huber_loss,
}


conf_pong_dqvmax_offline = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qfunc": conv_adam, "vfunc": conv_adam},
    "exploration": {"call_": exploration.egreedy, "epsilon_eval": 0.001},
    "agent": {
        "call_": DQVMaxAgent.DQVMaxAgent,
        "net_sync_freq": 2000,
        "train_freq": 4,
        "clip_rewards": True,
    },
    "memory": {
        "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
        "batch_size": 256,
    },
    "env": {
        "call_": patcher.create_atari_environment,
        "environment_name": "Pong",
        "version": "v5",
    },
    "runner": {
        "log_level": logging.DEBUG,
        "call_": FixedBatchRunner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": int(250e3),
            "eval_steps": int(125e3),
            "iterations": 200,
            "eval_period": 1,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 50,
            "collection_name": exp_name,
        },
        "aim": {"call_": reporter.AimReporter, "repo": str(config.data_dir)},
    },
}


def main():
    # sample 10 replay buffers, 20% of DQN replay dataset
    # (for max only like 1 or 2)
    conf = ...
