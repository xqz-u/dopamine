import os

import optax
from dopamine.jax import losses
from thesis import config, exploration, patcher
from thesis.agents import DQVMaxAgent
from thesis.memory import offline_memory
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner, runner

# TODO
# death by max_episode_steps?


conv_adam = {
    "model": {"call_": patcher.NatureDQNNetwork},
    # "model": {"hiddens": (512, 512)},
    "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
    "loss_metric": losses.huber_loss,
}


conf_pong_dqvmax_offline = lambda exp_name: {
    "experiment_name": exp_name,
    "nets": {"qfunc": conv_adam, "vfunc": conv_adam},
    "exploration": {"call_": exploration.egreedy, "epsilon_eval": 0.001},
    "agent": {
        "call_": DQVMaxAgent.DQVMaxAgent,
        "net_sync_freq": 2000,
        "train_freq": 4,
        # "clip_rewards": True,
        # "min_replay_history": 2000
    },
    "memory": {
        "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
        "batch_size": 256,
    },
    "env": {
        "call_": patcher.create_atari_environment,
        "environment_name": "Pong",
        "version": "v5",
        # "environment_name": "CartPole",
        # "version": "v1",
    },
    "runner": {
        "call_": FixedBatchRunner.FixedBatchRunner,
        "experiment": {
            "schedule": "train_and_eval",
            "seed": 4,
            "steps": int(250e3),
            "eval_steps": int(125e3),
            "iterations": 200,
            "eval_period": 1,
            # "steps": 700,
            # "eval_steps": 500,
            # "iterations": 20,
            # "eval_period": 2,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 50,
            "collection_name": exp_name,
        },
        "aim": {
            "call_": reporter.AimReporter,
            # "repo": str(config.data_dir),
            "repo": str(config.scratch_data_dir),
        },
    },
}

# sample 10 replay buffers, 20% of DQN replay dataset
# (for max only like 1 or 2)
conf = conf_pong_dqvmax_offline("test_pong_dqvmax_off")
redundancy = 5
expanded_confs = runner.add_redundancies(conf, redundancy)
pong_dqn_buffers_dir = os.path.join(config.data_dir, "Pong")
expanded_confs = runner.add_offline_buffers(
    expanded_confs,
    pong_dqn_buffers_dir,
    intermediate_dirs="replay_logs",
    iterations=[1],
)
conf_0, *rest_confs = expanded_confs
from thesis import utils

utils.data_dir_from_conf(
    conf_0["experiment_name"], conf_0, basedir=config.scratch_data_dir
)
run = runner.create_runner(conf_0)
# run.run_experiment()
