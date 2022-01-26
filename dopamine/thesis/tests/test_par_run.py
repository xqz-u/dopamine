import logging

from dopamine.discrete_domains import gym_lib
from dopamine.jax import losses
from thesis import exploration
from thesis.agents import dqv_max
from thesis.runner import parallel_runs, reporter

# repo = "/home/xqz-u/uni/thesis/resources/data/test_par_runner_1"
repo = "/home/xqz-u/dopamine/thesis/resources/data/test_par_runner_1"
cartpole_path = "CartPole-DQVMax"
acrobot_path = "Acrobot-DQVMax"
cartpole_preprocs = {
    "min_vals": tuple(gym_lib.CARTPOLE_MIN_VALS),
    "max_vals": tuple(gym_lib.CARTPOLE_MAX_VALS),
}
acrobot_preprocs = {
    "min_vals": tuple(gym_lib.ACROBOT_MIN_VALS),
    "max_vals": tuple(gym_lib.ACROBOT_MAX_VALS),
}

conf_cartpole = {
    "nets": {
        "qnet": {
            "model": {
                # "call_": ...,
                "hiddens": (512, 512),
                **cartpole_preprocs,
            },
            "optim": {
                # "call_": ...,
                "learning_rate": 0.001
            },
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {
                "hiddens": (512, 512),
                **cartpole_preprocs,
            },
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {"call_": exploration.egreedy_linear_decay},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e4),
        "observation_shape": (4, 1),
    },
    "env": {
        # "call_": gym_lib.create_gym_env,
        "environment_name": "CartPole",
        "version": "v0",
        # "clip_rewards": True
    },
    "memory": {
        # "call_": outofgraph...,
        # "stack_size": 1
    },
    "runner": {
        "base_dir": f"{repo}/{cartpole_path}",
        # "schedule": "continuous_train_and_eval",
        # "log_level": logging.INFO,
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
                "experiment": "CartPole-DQVMax",
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
                **acrobot_preprocs,
            },
            "optim": {
                # "call_": ...,
                "learning_rate": 0.001
            },
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {
                "hiddens": (512, 512),
                **acrobot_preprocs,
            },
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {"call_": exploration.egreedy_linear_decay},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e4),
        "observation_shape": (6, 1),
    },
    "env": {
        # "call_": gym_lib.create_gym_env,
        "environment_name": "Acrobot",
        "version": "v1",
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
                "experiment": "Acrobot-DQVMax",
            }
        ],
    },
}

# parallel_runs.run_multiple_configs([conf_cartpole, conf_acrobot], 2)
# parallel_runs.run_experiment(conf_cartpole)
# parallel_runs.run_experiment(conf_acrobot)
