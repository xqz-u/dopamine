import os

from thesis import config, offline_circular_replay_buffer, utils
from thesis.experiments import dqvmax_cartpole_growingbatch_train as dqvmax_gb
from thesis.runner import FixedBatchRunner, runner


# conf["runner"]["experiment"].update(
#     {
#         "termination_criterion": FixedBatchRunner.winning_termination,
#         "termination_args": {"min_return": 195},
#     }
# )
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
    utils.data_dir_from_conf(exp_name, conf)
    run = runner.create_runner(conf)
    return run
    # run.run_experiment_with_redundancy()


x = main()
# x.run_experiment_with_redundancy()
# import pprint

# pprint.pprint(x.hparams)

# main()
