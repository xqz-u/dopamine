from thesis.memory import prio_offline_memory
from thesis.runner import runner
from thesis.scratch import test_off_mem

conf = test_off_mem.conf_cartpole_dqvmax_offline("test_cp_dqvmax_off_prio")

conf["memory"]["call_"] = prio_offline_memory.PrioritizedOfflineOutOfGraphReplayBuffer

all_confs = test_off_mem.main(conf)


# from thesis import config, utils

# myconf = all_confs[0]
# utils.data_dir_from_conf(
#     myconf["experiment_name"], myconf, basedir=config.scratch_data_dir
# )
# run = runner.create_runner(myconf)
# run.run_experiment()


# runner.run_experiment_atomic(all_confs[0], scratch=True)

# runner.p_run_experiments(all_confs, scratch=True)
