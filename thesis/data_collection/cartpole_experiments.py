#!/usr/bin/env python3

import os

import gin

from dopamine.discrete_domains import run_experiment
from thesis import config


def run(conf_path):
    gin.parse_config_file(conf_path)
    runner = run_experiment.create_runner()
    runner.run_experiment()


gin.enter_interactive_mode()

configs_path = os.path.join(config.base_dir, "data_collection", "configs")

dqn_cartpole_classic = os.path.join(configs_path, "dqn_cartpole.gin")
dqv_cartpole_classic = os.path.join(configs_path, "dqv_cartpole.gin")
dqv_max_cartpole_classic = os.path.join(configs_path, "dqv_max_cartpole.gin")
dqv_max_cartpole_classic_offline = os.path.join(
    configs_path, "dqv_max_cartpole_offline.gin"
)

# run(dqn_cartpole_classic)
# run(dqv_cartpole_classic)
# run(dqv_max_cartpole_classic)
run(dqv_max_cartpole_classic_offline)
