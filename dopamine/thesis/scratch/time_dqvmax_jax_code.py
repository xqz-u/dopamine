import time

from thesis import config, utils
from thesis.experiments import pg_time_train_iter_cc
from thesis.runner import runner

if __name__ == "__main__":
    conf = pg_time_train_iter_cc.make_conf("time_dqvmax_jax_train")
    conf, *_ = pg_time_train_iter_cc.doconfs(conf, config.data_dir)
    conf["reporters"]["aim"]["repo"] = str(config.scratch_data_dir)
    utils.data_dir_from_conf(
        conf["experiment_name"], conf, basedir=config.scratch_data_dir
    )
    # conf, *_ = pg_time_train_iter_cc.doconfs(conf, config.peregrine_data_dir)
    # conf["reporters"]["aim"]["repo"] = str(config.peregrine_data_dir)
    # utils.data_dir_from_conf(
    #     conf["experiment_name"], conf, basedir=config.peregrine_data_dir
    # )
    run = runner.create_runner(conf)
    replay_elts = run.agent.sample_memory()

    start = time.time()
    run.agent.train_v(replay_elts)
    print("vtrain_time: {time.time() - start}")

    start = time.time()
    run.agent.train_q(replay_elts)
    print("qtrain_time: {time.time() - start}")
