import time

from thesis import constants, utils
from thesis.experiments import pg_time_train_iter_cc
from thesis.runner import runner

if __name__ == "__main__":
    conf = pg_time_train_iter_cc.make_conf("time_only_train_cc")
    # conf, *_ = pg_time_train_iter_cc.doconfs(conf, constants.data_dir)
    # conf["reporters"]["aim"]["repo"] = str(constants.scratch_data_dir)
    # utils.data_dir_from_conf(
    #     conf["experiment_name"], conf, basedir=constants.scratch_data_dir
    # )
    conf, *_ = pg_time_train_iter_cc.doconfs(conf, constants.peregrine_data_dir)
    conf["reporters"]["aim"]["repo"] = str(constants.peregrine_data_dir)
    utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=constants.peregrine_data_dir
    )
    run = runner.create_runner(conf)
    start = time.time()
    for _ in range(run.steps):
        run.agent.learn()
    print(f"dqvmax learn exec time: {time.time() - start}")
