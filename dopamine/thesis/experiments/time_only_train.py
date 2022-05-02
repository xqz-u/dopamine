import time

from thesis import config, utils
from thesis.experiments import peregrine_time_train_iter
from thesis.runner import runner

conf = peregrine_time_train_iter.make_conf("cip")
conf, *_ = peregrine_time_train_iter.doconfs(conf, config.data_dir)
conf["reporters"]["aim"]["repo"] = str(config.scratch_data_dir)
utils.data_dir_from_conf(conf["experiment_name"], conf, basedir=config.scratch_data_dir)
run = runner.create_runner(conf)
start = time.time()
for _ in range(run.steps):
    run.agent.learn()
print(f"dqvmax learn exec time: {time.time() - start}")
