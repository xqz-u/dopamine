import logging

from dopamine.jax import losses
from thesis import exploration
from thesis.agents import dqv_max
from thesis.runner import parallel_runs, reporter

# repo = "/home/xqz-u/uni/thesis/resources/data/test_par_runner_1"
repo = "/home/xqz-u/uni/dopamine/resources/data/test_par_runner_1"
cartpole_path = "CartPole-DQVMax"
acrobot_path = "Acrobot-DQVMax"

conf_cartpole = {
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
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e4),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "memory": {},
    "runner": {
        "base_dir": f"{repo}/{cartpole_path}",
        # "schedule": "continuous_train_and_eval",
        "log_level": logging.INFO,
        # "resume": False,
        "experiment": {
            "seed": 4,
            "steps": 1000,
            "iterations": 1000,
            "redundancy": 5,
        },
        "reporters": [
            {
                "call_": reporter.AimReporter,
                "repo": repo,
                "experiment": cartpole_path,
            }
        ],
    },
}


conf_acrobot = {
    "nets": {
        "qnet": {
            "model": {
                # "call_": ...,
                "hiddens": (512, 512),
            },
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {
                "hiddens": (512, 512),
            },
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e4),
    },
    "env": {
        # "call_": gym_lib.create_gym_env,
        "environment_name": "Acrobot",
        "version": "v1"
        # "clip_rewards": True
    },
    "memory": {
        # "call_": outofgraph...,
        # "stack_size": 1
    },
    "runner": {
        "base_dir": f"{repo}/{acrobot_path}",
        # "schedule": "continuous_train_and_eval",
        "log_level": logging.INFO,
        # "resume": False,
        "experiment": {
            "seed": 10,
            "steps": 1000,
            "iterations": 1000,
            "redundancy": 5,
        },
        "reporters": [
            {
                "call_": reporter.AimReporter,
                "repo": repo,
                "experiment": acrobot_path,
            }
        ],
    },
}


parallel_runs.run_multiple_configs([conf_cartpole, conf_acrobot])
# parallel_runs.run_experiment(conf_cartpole)
# parallel_runs.run_experiment(conf_acrobot)
