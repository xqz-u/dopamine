import inspect
import logging
import pprint

from dopamine.jax import losses
from thesis import exploration
from thesis.agents import dqv_max
from thesis.runner import reporter, runner

test_path = "/home/xqz-u/uni/thesis/resources/data/test_runner"
# test_path = "/home/xqz-u/uni/dopamine/resources/data/test_runner"

conf = {
    "nets": {
        "qnet": {
            "model": {
                # "call_": ...,
                "hiddens": (512, 512),
            },
            "optim": {
                # "call_": ...,
                "learning_rate": 0.001
            },
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    # "exploration": {"call_": exploration.egreedy},
    "exploration": {},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(1e4),
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
        "base_dir": test_path,
        # "schedule": "continuous_train_and_eval",
        # "log_level": logging.INFO,
        # "resume": False,
        "experiment": {
            # "seed": 4,
            "steps": 2000,
            "iterations": 10,
            "redundancy": 3,
        },
        "reporters": [
            {
                "call_": reporter.AimReporter,
                "repo": test_path,
                "experiment": "test_runner",
            }
        ],
    },
}


runner = runner.create_runner(conf)
# pprint.pprint(runner.hparams)
runner.run_experiment_with_redundancy()

# runner.reporters[0].setup(0)


# runner.reporters[0]([("Return", 2.0)], 1)

# runner.run_one_episode("train")


# import aim

# repo = aim.Repo("/home/xqz-u/uni/thesis/resources/data/test_runner")

# df = []
# for run_iteration in repo.query_metrics("metric.name == 'return'").iter_runs():
#     df.append(run_iteration.dataframe())
