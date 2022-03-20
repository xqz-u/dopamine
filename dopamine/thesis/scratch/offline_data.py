import os

from thesis import config, offline_circular_replay_buffer
from thesis.runner import runner

dqn_logdir = os.path.join(
    config.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
)
dqn_exp_conf = {
    "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    "checkpoint_dir": dqn_logdir,
    "iterations": list(range(496, 500)),
}


def offline_runner(conf):
    conf["runner"]["call_"] = runner.FixedBatchRunner
    conf["memory"] = dqn_exp_conf
