import logging

from dopamine.jax import losses
from thesis.agents import dqv_max
from thesis.runner import reporter, runner

# repo = "/home/xqz-u/uni/thesis/resources/data/offline_runner"
# repo = "/home/xqz-u/uni/thesis/resources/data/train_cartpole_online"
# repo = "/home/xqz-u/uni/thesis/resources/data/eval_cartpole_online"
repo = "/home/xqz-u/uni/thesis/resources/data/train_cartpole_offline"
# repo = "/home/xqz-u/uni/thesis/resources/data/eval_cartpole_offline"
cartpole_path = "CartPole-DQVMax"

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
        "min_replay_history": int(5e3),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "memory": {"replay_capacity": int(5e4)},
    "runner": {
        "call_": runner.GrowingBatchRunner,
        "base_dir": f"{repo}/{cartpole_path}",
        # "schedule": "train",
        # "schedule": "eval",
        # 'logging_file_prefix': "pippo",
        # 'ckpt_file_prefix': "pippo",
        "log_level": logging.DEBUG,
        "experiment": {
            "seed": 4,
            "steps": 600,
            "iterations": 1000,
            "redundancy": 3,
            # "fitting_steps": 599,
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

run = runner.create_runner(conf_cartpole)
run.run_experiment_with_redundancy()

# runner.run_multiple_configs([conf_cartpole])

# run = runner.create_runner(conf_cartpole)(
#     conf_cartpole, **conf_cartpole["runner"]["experiment"]
# )
