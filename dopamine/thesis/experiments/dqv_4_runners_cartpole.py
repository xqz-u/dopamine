import logging
import os

import optax
from dopamine.jax import losses
from thesis import config, offline_circular_replay_buffer
from thesis.agents import DQVAgent
from thesis.runner import reporter, runner

dqn_logdir = os.path.join(
    config.data_dir, "CartPole-v0", "JaxDQNAgent", "online_train", "checkpoints"
)
dqn_exp_conf = {
    "call_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    "checkpoint_dir": dqn_logdir,
    "iterations": list(range(496, 500)),
}


model_conf = {
    "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001},
    "loss": losses.huber_loss,
}
make_config = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qnet": model_conf, "vnet": model_conf},
    "exploration": {},
    "agent": {
        "call_": DQVAgent.DQVAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e3),
    },
    "env": {"environment_name": "CartPole", "version": "v0"},
    # "memory": {"replay_capacity": int(5e4)},
    "runner": {
        # "call_": runner.GrowingBatchRunner,
        "schedule": "train",
        "log_level": logging.INFO,
        "experiment": {
            "seed": 4,
            "steps": 600,
            "iterations": 1000,
            "redundancy": 3,
        },
        "reporters": [{"call_": reporter.AimReporter, "repo": str(config.aim_dir)}],
    },
}


def online_conf():
    conf = make_config("dqv_train")
    conf["memory"] = {"replay_capacity": int(5e4)}
    return conf


def gb_conf():
    conf = make_config("dqv_gb_train")
    conf["runner"]["call_"] = runner.GrowingBatchRunner
    conf["memory"] = {"replay_capacity": int(5e4)}
    return conf


def gb_dqn_exp_conf():
    conf = make_config("dqv_gb_dqn_exp_train")
    conf["runner"]["call_"] = runner.GrowingBatchRunner
    conf["memory"] = dqn_exp_conf
    return conf


def fb_dqn_exp_conf():
    conf = make_config("dqv_fb_dqn_exp_train")
    conf["runner"]["call_"] = runner.FixedBatchRunner
    conf["memory"] = dqn_exp_conf
    return conf


def main():
    all_configs = [
        conf_maker()
        for conf_maker in [online_conf, gb_conf, gb_dqn_exp_conf, fb_dqn_exp_conf]
    ]
    runner.run_multiple_configs(all_configs)


main()
