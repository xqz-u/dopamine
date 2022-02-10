import logging

from dopamine.jax import losses
from thesis.agents import dqv_max
from thesis.runner import reporter, runner

# repo = "/home/xqz-u/uni/thesis/resources/data/offline_runner"
# repo = "/home/xqz-u/uni/dopamine/resources/data/train_cartpole_online"
# repo = "/home/xqz-u/uni/dopamine/resources/data/eval_cartpole_online"
repo = "/home/xqz-u/uni/dopamine/resources/data/train_cartpole_offline"
# repo = "/home/xqz-u/uni/dopamine/resources/data/eval_cartpole_offline"
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
        "base_dir": f"{repo}/{cartpole_path}",
        # "rl_mode": "online",
        "rl_mode": "offline",
        "schedule": "train",
        # "schedule": "continuous_train_and_eval",
        # 'logging_file_prefix': "pippo",
        # 'ckpt_file_prefix': "pippo",
        "log_level": logging.DEBUG,
        "experiment": {
            "seed": 4,
            "steps": 600,
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


runner.run_multiple_configs([conf_cartpole])

# run = runner.create_runner(conf_cartpole)(
#     conf_cartpole, **conf_cartpole["runner"]["experiment"]
# )
