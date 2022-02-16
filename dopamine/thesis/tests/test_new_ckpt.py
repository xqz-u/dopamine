import os

from thesis import config, utils
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import OnlineRunner, runner

exp_name = "pippo"
test_data_dir = os.path.join(config.data_dir, "tests")

conf = dqvmax_gb.make_config(exp_name)
conf["runner"]["call_"] = OnlineRunner.OnlineRunner
conf["runner"]["experiment"]["iterations"] = 15

utils.data_dir_from_conf(exp_name, conf, basedir=test_data_dir)

run = runner.create_runner(conf)
run.run_experiment_with_redundancy()
