import logging
import os

import optax
from dopamine.jax import losses
from thesis import config
from thesis.agents import DQVMaxAgent
from thesis.memory import offline_memory
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner, runner

mlp_adam = lambda name: {
    name: {
        "model": {"hiddens": (512, 512)},
        "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
        "loss_metric": losses.huber_loss,
    }
}

conf_cartpole_dqvmax_offline = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {**mlp_adam("qfunc"), **mlp_adam("vfunc")},
    "exploration": {},
    "agent": {"call_": DQVMaxAgent.DQVMaxAgent, "net_sync_freq": int(1e4)},
    "memory": {
        "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
        "batch_size": 256,
    },
    "env": {"environment_name": "CartPole", "version": "v1"},
    "runner": {
        "log_level": logging.DEBUG,
        "call_": FixedBatchRunner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": 1000,
            "iterations": 100,
            "eval_period": 3,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 25,
            "collection_name": exp_name,
        },
        "aim": {"call_": reporter.AimReporter, "repo": str(config.scratch_data_dir)},
    },
}


def main(exp_name: str):
    conf = conf_cartpole_dqvmax_offline(exp_name)
    dqna_cartpole_buffers_dir = os.path.join(
        config.data_dir,
        "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
    )
    repeat = 3
    buffers_iterations = None
    # buffers_iterations = [[1, 2, 3], [2]]
    conf, *rest_confs = runner.add_offline_buffers(
        runner.add_redundancies(conf, repeat),
        dqna_cartpole_buffers_dir,
        iterations=buffers_iterations,
    )
    all_confs = [conf, *rest_confs]
    return all_confs
    # for c in all_confs:
    #     pprint(c["memory"])
    # runner.run_experiments(all_confs)
    # from thesis import utils

    # utils.data_dir_from_conf(conf["experiment_name"], conf)
    # run = runner.create_runner(conf)


runner.p_run_experiments(main("cp_dqvmax_offline_test"), scratch=True)
