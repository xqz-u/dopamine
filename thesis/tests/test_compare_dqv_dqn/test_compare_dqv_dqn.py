#!/usr/bin/env python3

import collections
import os

import gin

from dopamine.colab import utils as colab_u
from dopamine.discrete_domains import run_experiment
from thesis import utils as u

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
HERE = os.path.join(BASE_DIR, "thesis/tests/test_compare_dqv_dqn")
dqn_data_dir = os.path.join(HERE, "dqn_data")
dqv_data_dir = os.path.join(HERE, "dqv_data")


@u.timer
def run_timed(runner):
    runner.run_experiment()


def start(gin_conf_file, data_dir):
    cartpole_conf = os.path.join(HERE, gin_conf_file)
    gin.parse_config_file(cartpole_conf)
    # run_timed(run_experiment.create_runner(data_dir))
    return run_experiment.create_runner(data_dir)


def main():
    start("dqv_cartpole.gin", dqn_data_dir)
    # print("sssssssssssssssssssssssssssssssssssssssssssssssssssssss END FIRST ACTOR")
    # start("dqv_cartpole.gin", dqv_data_dir)
    return colab_u.read_experiment(
        HERE,
        # parameter_set=collections.OrderedDict([("agent", ["dqn", "dqv"])]),
        parameter_set=collections.OrderedDict([("agent", ["dqn"])]),
        job_descriptor="{}_data",
        summary_keys=["train_episode_returns"],
        verbose=True,
    )


# stats = main()
