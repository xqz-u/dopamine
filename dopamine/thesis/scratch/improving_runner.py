import logging
import os

import optax
from dopamine.jax import losses
from thesis import config, offline_circular_replay_buffer, utils
from thesis.agents import DQVMaxAgent
from thesis.reporter import reporter
from thesis.runner import runner

dqn_logdir = os.path.join(
    config.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
)
dqn_exp_conf = {
    "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    "checkpoint_dir": dqn_logdir,
    "iterations": list(range(496, 500)),
}


model_conf = {
    "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss": losses.huber_loss,
}
make_config = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qfunc": model_conf, "vfunc": model_conf},
    "exploration": {},
    "agent": {
        "call_": DQVMaxAgent.DQVMaxAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e2),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    "runner": {
        "log_level": logging.DEBUG,
        "experiment": {
            "schedule": "train",
            "seed": 4,
            "steps": 600,
            "iterations": 3,
            "redundancy": 2,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "db_name": "test_database",
            "buffering": 3,
        },
        # "aim": {"call_": reporter.AimReporter},
    },
}


def online_runner(conf):
    conf["memory"] = {"replay_capacity": int(5e4)}


def offline_runner(conf):
    conf["runner"]["call_"] = runner.FixedBatchRunner
    conf["memory"] = dqn_exp_conf


exp_name = "testone"
conf = make_config(exp_name)
# data_dir = utils.data_dir_from_conf(exp_name, conf, basedir=config.scratch_data_dir)
# # conf["reporters"]["aim"]["repo"] = data_dir
# conf["reporters"]["mongo"]["collection_name"] = exp_name
# online_runner(conf)
# # offline_runner(conf)
# run = runner.create_runner(conf)
# run.run_experiment_with_redundancy()
# # self = run
# # conf_id = id(conf)
