import logging
import os

import gin
import optax
from dopamine.discrete_domains import gym_lib, run_experiment
from dopamine.jax import losses
from thesis import config, networks, offline_circular_replay_buffer, patcher, utils
from thesis.agents import DQNAgent
from thesis.reporter import reporter
from thesis.runner import FixedBatchRunner, runner


def make_env_conf(env_name: str, version: str, is_atari: bool = False) -> dict:
    return {
        "call_": gym_lib.create_gym_environment
        if not is_atari
        else patcher.create_atari_environment,
        "environment_name": env_name,
        "version": version,
    }


make_config = lambda exp_name, env, version, is_atari=False: {
    "experiment_name": exp_name,
    "nets": {
        "qfunc": {
            "model": {"hiddens": (512, 512)},
            "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
            "loss_metric": losses.huber_loss,
        }
    },
    "exploration": {},
    "agent": {
        "call_": DQNAgent.DQNAgent,
        "net_sync_freq": int(1e4),
        "min_replay_history": int(5e3),
    },
    "memory": {"replay_capacity": int(5e4)},
    "env": make_env_conf(env, version, is_atari),
    "runner": {
        "log_level": logging.DEBUG,
        "experiment": {
            "schedule": "train",
            "seed": 4,
            "steps": 1000,
            "iterations": 3,
        },
    },
    "reporters": {
        "mongo": {
            "call_": reporter.MongoReporter,
            "buffering": 25,
            "collection_name": exp_name,
        },
        "aim": {"call_": reporter.AimReporter, "repo": str(config.data_dir)},
    },
}


def pong_dopamine():
    pong_gin_conf = os.path.join(
        config.dopamine_dir, "jax/agents/dqn/configs/dqn_profiling.gin"
    )
    data_path = os.path.join(config.scratch_data_dir, "Pong", "dqn")
    gin.parse_config_file(pong_gin_conf)
    run = run_experiment.create_runner(data_path)
    run.run_experiment()


def cp(render_mode="rgb_array"):
    cp_conf = make_config("cp_new", "CartPole", "v1")
    cp_conf["env"]["render_mode"] = render_mode
    cp_conf["runner"]["experiment"]["redundancy_nr"] = 0
    cp_conf["runner"]["base_dir"] = os.path.join(
        config.scratch_data_dir, "CartPole-v1/DQNAgent/cp_new"
    )
    return runner.create_runner(cp_conf)


def pong(render_mode: str = "rgb_array"):
    pong_conf = make_config("test_pong", "ALE/Pong", "v5", True)
    pong_conf["nets"]["qfunc"]["model"] = {"call_": networks.NatureDQNNetwork}
    pong_conf["env"]["render_mode"] = render_mode
    pong_conf["runner"]["experiment"]["redundancy_nr"] = 0
    pong_conf["runner"]["base_dir"] = utils.data_dir_from_conf(
        pong_conf["experiment_name"], pong_conf, basedir=config.scratch_data_dir
    )
    run_pong = runner.create_runner(pong_conf)
    return run_pong


# run_cp = cp("human")
# run_cp.run_experiment()

# run_pong = pong("human")
# run_pong.run_experiment()
