import logging

from thesis import config, utils
from thesis.agents.DQVAgent import DQVAgent
from thesis.experiments import experiment_all_runners
from thesis.runner import runner

conf = experiment_all_runners.online_conf()
conf["agent"]["call_"] = DQVAgent
conf["runner"]["log_level"] = logging.DEBUG
conf["experiment_name"] = "dqn_online_train"
utils.data_dir_from_conf(conf["experiment_name"], conf, str(config.test_dir))
run = runner.create_runner(conf)
run.run_experiment_with_redundancy()
