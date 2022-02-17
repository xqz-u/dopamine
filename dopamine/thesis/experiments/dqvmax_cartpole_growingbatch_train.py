import logging

import optax
from dopamine.jax import losses
from thesis import config
from thesis.agents.DQVMaxAgent import DQVMaxAgent
from thesis.runner import reporter, runner

model_conf = {
    "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001},
    "loss": losses.huber_loss,
}

make_config = lambda exp_name: {
    "experiment_name": f"dqvmax_{exp_name}",
    "nets": {"qnet": model_conf, "vnet": model_conf},
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
