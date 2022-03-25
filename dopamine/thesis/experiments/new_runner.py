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


def main():
    conf = make_config("test_new_runner_nofull", "CartPole", "v1")
    # conf["runner"]["experiment"]["record_experience"] = False
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    run = runner.create_runner(conf)
    run.run_experiment()


def test_single_run():
    conf = make_config("test_new_runner_single", "CartPole", "v1")
    # return conf
    runner.run_experiment_atomic(conf)


def test_redundancies():
    conf = make_config("test_new_runner_redund", "CartPole", "v1")
    runner.p_run_experiment(conf, repeat=3)


def test_redundancies_serial():
    conf = make_config("test_new_runner_serial", "CartPole", "v1")
    runner.run_experiment(conf, repeat=3)


def test_mul_confs():
    conf_cp = make_config("test_new_runner_nofull_cp", "CartPole", "v1")
    conf_ab = make_config("test_new_runner_nofull_ab", "Acrobot", "v1")
    runner.p_run_multiple_configs([conf_cp, conf_ab])


# main()
# test_single_run()
# test_redundancies_serial()
# test_redundancies()

# test_mul_confs()
# c = test_single_run()
# utils.data_dir_from_conf(c["experiment_name"], c)
# run = runner.create_runner(c)

# import multiprocessing as mp
# import signal
# import time

# from thesis.runner import runner


# def f(el):
#     runner.mp_print("sleep 3 seconds...")
#     time.sleep(3)
#     runner.mp_print("DONE sleeping")
#     return el * 2


# def handler(signum, frame):
#     runner.mp_print(f"handle signal {signum}")
#     raise (KeyboardInterrupt)


# signal.signal(signal.SIGINT, handler)

# procs = 3
# with mp.Pool(processes=procs) as p:
#     res = p.map(f, range(procs * 10))
# print(res)
