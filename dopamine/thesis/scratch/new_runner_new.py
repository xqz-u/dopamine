import logging

import optax
from dopamine.jax import losses
from thesis import config
from thesis.agents import DQNAgent
from thesis.experiments import cp_ab_dqvmax_distr_shift
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
        # "min_replay_history": int(5e3),
        "min_replay_history": 500,
    },
    # "memory": {"replay_capacity": int(5e4)},
    "memory": {"replay_capacity": 700},
    "env": {"environment_name": env, "version": version},
    "runner": {
        "log_level": logging.DEBUG,
        "experiment": {
            "schedule": "train",
            # "record_experience": True,
            "seed": 4,
            "steps": 1000,
            "iterations": 3,
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


def test_serial():
    runner.run_experiments(
        [
            (make_config("test_new_runner_nofull_cp_s", "CartPole", "v1"), 3),
            (make_config("test_new_runner_nofull_ab_s", "Acrobot", "v1"), 3),
        ]
    )


def test_parallel():
    runner.p_run_experiments(
        [
            (make_config("test_new_runner_nofull_cp", "CartPole", "v1"), 3),
            (make_config("test_new_runner_nofull_ab", "Acrobot", "v1"), 3),
        ]
    )


def offline_cp_ab_confs():
    def do_conf(exp_name, env, version, repeats, off_exp_name):
        c = make_config(exp_name, env, version, repeats)
        c["memory"] = cp_ab_dqvmax_distr_shift.dqn_full_exp_conf(
            off_exp_name, f"{env}-{version}"
        )
        return c

    return [
        do_conf(
            "test_cp_full_offline", "CartPole", "v1", 3, "cp_dqn_full_experience_%%"
        ),
        do_conf(
            "test_ab_full_offline", "Acrobot", "v1", 3, "ab_dqn_full_experience_%%"
        ),
    ]


def test_serial_offline():
    runner.run_experiments(offline_cp_ab_confs)


def test_parallel_offline():
    runner.p_run_experiments(offline_cp_ab_confs)


# test_serial()
# test_parallel()
# test_serial_offline()
# test_parallel_offline()
