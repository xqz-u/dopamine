import logging

import optax
from dopamine.jax import losses
from thesis import config
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
            "steps": 1000,
            "iterations": 500,
            "redundancy": 3,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 25,
            "collection_name": exp_name,
        },
        "aim": {"call_": reporter.AimReporter, "repo": str(config.data_dir)},
    },
}


def main():
    runner.run_multiple_configs(
        [
            make_config(*c)
            for c in [
                ("cp_dqn_full_experience_%%", "CartPole", "v1"),
                ("ab_dqn_full_experience_%%", "Acrobot", "v1"),
            ]
        ]
    )


def main_cp():
    cp_conf = make_config("cp_dqn_full_experience_%%", "CartPole", "v1")
    runner.run_experiment([cp_conf, 0])


def main_ab():
    ab_conf = make_config("ab_dqn_full_experience_%%", "Acrobot", "v1")
    runner.run_experiment([ab_conf, 0])


# if __name__ == "__main__":
# main_cp()
# main_ab()
