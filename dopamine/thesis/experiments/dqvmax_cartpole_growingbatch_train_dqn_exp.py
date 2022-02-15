import os

from thesis import config, offline_circular_replay_buffer, utils
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import runner


def main():
    exp_name = "growingbatch_train_dqn_experience"
    conf = dqvmax_gb.make_config(exp_name)
    # TODO iterations
    conf["memory"].update(
        {
            "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
            "checkpoint_dir": os.path.join(
                config.data_dir,
                "CartPole-v0",
                "JaxDQNAgent",
                "online_train",
                "checkpoints",
            ),
        }
    )
    utils.data_dir_from_conf(exp_name, dqvmax_gb.conf)
    run = runner.create_runner(dqvmax_gb.conf)
    run.run_experiment_with_redundancy()
