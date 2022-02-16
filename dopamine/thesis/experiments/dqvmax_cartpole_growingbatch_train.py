import logging

from dopamine.jax import losses
from thesis import config
from thesis.agents.DQVMaxAgent import DQVMaxAgent
from thesis.runner import reporter, runner

make_config = lambda exp_name: {
    "experiment_name": f"dqvmax_{exp_name}",
    "nets": {
        "qnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {},
    "agent": {
        "call_": DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e3),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "memory": {"replay_capacity": int(5e4)},
    "runner": {
        "call_": runner.GrowingBatchRunner,
        "schedule": "train",
        "log_level": logging.INFO,
        "experiment": {
            "seed": 4,
            "steps": 600,
            "iterations": 1000,
            "redundancy": 3,
        },
        "reporters": [{"call_": reporter.AimReporter, "repo": config.aim_dir}],
    },
}


def main():
    exp_name = "growingbatch_train"
    conf = make_config(exp_name)
    runner.run_multiple_configs([conf])


# main()
