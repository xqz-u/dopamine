import os

from thesis import config, offline_circular_replay_buffer
from thesis.agents import DQVMaxAgent
from thesis.experiments.cp_ab_dqn_full_experience import make_config
from thesis.runner import runner

dqn_full_exp_conf = lambda exp_name, full_env: {
    "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    "_buffers_root_dir": os.path.join(
        config.data_dir, f"{full_env}/DQNAgent/{exp_name}/checkpoints/full_experience"
    ),
}


def main(exp_name, env, version, full_experience_exp_name):
    conf = make_config(exp_name, env, version)
    conf["nets"]["vfunc"] = conf["nets"]["qfunc"]
    conf["agent"]["call_"] = DQVMaxAgent.DQVMaxAgent
    conf["runner"]["call_"] = runner.FixedBatchRunner
    conf["memory"] = dqn_full_exp_conf(full_experience_exp_name, f"{env}-{version}")
    conf["runner"]["experiment"].pop("record_experience", None)
    conf["runner"]["experiment"]["schedule"] = "train_and_eval"
    conf["runner"]["experiment"]["eval_period"] = 2
    runner.run_experiment([conf, 0])
    # conf["runner"]["experiment"]["iterations"] = 2
    # from thesis import utils

    # utils.data_dir_from_conf(exp_name, conf)
    # return runner.create_runner(conf)


if __name__ == "__main__":
    pass
    # main("cp_dqvmax_distr_shift", "CartPole", "v1", "cp_dqn_full_experience_%%")
    # main("ab_dqvmax_distr_shift", "Acrobot", "v1", "ab_dqn_full_experience_%%")
