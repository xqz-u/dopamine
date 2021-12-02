#!/usr/bin/env python3

import os

import gin

from dopamine.discrete_domains import run_experiment

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
HERE = os.path.join(BASE_DIR, "thesis/tests/dqv_cartpole_train")

cartpole_conf = os.path.join(HERE, "dqv_cartpole_train.gin")
gin.parse_config_file(cartpole_conf)

runner = run_experiment.create_runner(os.path.join(HERE, "data"))
runner.run_experiment()
