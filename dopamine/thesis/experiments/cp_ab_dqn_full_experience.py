import logging

import optax
from dopamine.jax import losses
from thesis import config, utils
from thesis.agents import DQNAgent
from thesis.reporter import reporter
from thesis.runner import runner

make_config = lambda exp_name, env, version: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": {
            "model": {"hiddens": (512, 512)},
            "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
            "loss_metric": losses.huber_loss,
        }
    },
    "exploration": {},
    "agent": {
        "call_": DQNAgent.DQNAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e3),
    },
    "memory": {"replay_capacity": int(5e4)},
    "env": {"environment_name": env, "version": version},
    "runner": {
        "log_level": logging.DEBUG,
        "experiment": {
            "schedule": "train",
            "record_experience": True,
            "seed": 4,
            "steps": int(1e3),
            "iterations": 50,
            "redundancy": 2,
        },
    },
    "reporters": {
        "mongo": {"call_": reporter.MongoReporter, "buffering": 25},
        "aim": {"call_": reporter.AimReporter, "repo": str(config.data_dir)},
    },
}


def make_configs():
    def with_dir(name, *args):
        conf = make_config(name, *args)
        utils.data_dir_from_conf(conf["experiment_name"], conf)
        return conf

    return [
        with_dir("cp_dqn_full_experience", "CartPole", "v1"),
        with_dir("ab_dqn_full_experience", "Acrobot", "v1"),
    ]


def main():
    runner.run_multiple_configs(make_configs())


# main()
