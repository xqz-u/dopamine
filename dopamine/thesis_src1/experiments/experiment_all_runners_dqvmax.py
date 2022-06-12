import os

from thesis import constants, offline_circular_replay_buffer
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import runner

# NOTE this won't run, use configs from thesis.experiments.dqv_4_runners_cartpole


def online_conf():
    conf = dqvmax_gb.make_config("online_train")
    conf["runner"]["call_"] = runner.OnlineRunner
    return conf


def gb_conf():
    return dqvmax_gb.make_config("growingbatch_train")


def gb_dqn_exp_conf():
    dqn_logdir = os.path.join(
        constants.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
    )
    conf = dqvmax_gb.make_config("growingbatch_train_dqn_experience")
    conf["memory"] = {
        "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
        "checkpoint_dir": dqn_logdir,
        "iterations": list(range(496, 500)),
    }
    return conf


def fb_dqn_exp_conf():
    dqn_logdir = os.path.join(
        constants.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
    )
    conf = dqvmax_gb.make_config("fixedbatch_train_dqn_experience")
    conf["runner"]["call_"] = runner.FixedBatchRunner
    conf["memory"] = {
        "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
        "checkpoint_dir": dqn_logdir,
        "iterations": list(range(496, 500)),
    }
    return conf


def main():
    all_configs = [
        conf_maker()
        for conf_maker in [online_conf, gb_conf, gb_dqn_exp_conf, fb_dqn_exp_conf]
    ]
    runner.run_multiple_configs(all_configs)


# main()
