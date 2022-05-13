from thesis import config
from thesis.experiments import pg_time_train_iter_cc
from thesis.runner import runner

conf = pg_time_train_iter_cc.make_conf("pippo")

conf, *_ = pg_time_train_iter_cc.doconfs(conf, config.data_dir)
run = runner.build_runner(conf, config.scratch_data_dir)
ag = run.agent

ag.train(ag.sample_memory())
