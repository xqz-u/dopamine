#!/usr/bin/env python3

import os

import gin

from dopamine.discrete_domains import run_experiment
from thesis import config

dqn_cartpole_conf = os.path.join(
    config.base_dir, "online", "data_collection", "configs", "dqn_cartpole.gin"
)
gin.parse_config_file(dqn_cartpole_conf)

runner = run_experiment.create_runner()
runner.run_experiment()
