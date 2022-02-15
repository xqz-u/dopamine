import os

from thesis import config, offline_circular_replay_buffer, utils
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import runner


# TODO problem: OfflineOutOfGraphReplayBuffer calls memory_gen_filename
# in patcher.py before a redundancy value is set (done by the runner
# itself) and so it fails. In general, probably need a better redesign
# of the OfflineOutOfGraphReplayBuffer...
def main():
    exp_name = "growingbatch_train_dqn_experience"
    dqn_logdir = os.path.join(
        config.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
    )
    conf = dqvmax_gb.make_config(exp_name)
    conf["memory"] = {
        "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
        "checkpoint_dir": dqn_logdir,
        "iterations": list(range(496, 500)),
    }
    utils.data_dir_from_conf(exp_name, conf)
    run = runner.create_runner(conf)
    run.run_experiment_with_redundancy()
