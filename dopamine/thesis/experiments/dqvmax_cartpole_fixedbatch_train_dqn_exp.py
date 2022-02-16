import os

from thesis import config, offline_circular_replay_buffer
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import FixedBatchRunner, runner


def main():
    exp_name = "fixedbatch_train_dqn_experience"
    dqn_logdir = os.path.join(
        config.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
    )
    conf = dqvmax_gb.make_config(exp_name)
    conf["runner"]["call_"] = FixedBatchRunner.FixedBatchRunner
    conf["memory"] = {
        "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
        "checkpoint_dir": dqn_logdir,
        "iterations": list(range(496, 500)),
    }
    conf["runner"]["experiment"]["eval_period"] = 1000
    runner.run_multiple_configs([conf])


main()
