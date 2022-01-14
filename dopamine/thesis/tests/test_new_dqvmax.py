import time

import aim
import gym
import jax
from dopamine.jax import losses
from thesis import exploration
from thesis.agents import dqv_max
from thesis.tests import simple_runner as sr

conf = {
    "nets": {
        "qnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    "exploration": {
        "fn": exploration.egreedy,
    },
    "memory": {"stack_size": 1},
    # "memory": {
    #     "class_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    #     "stack_size": 1,
    #     "checkpoint_dir": "/home/xqz-u/uni/dopamine/resources/data/JaxDQNAgent_CartPole-v0_ClassicControlDQNNetwork_ref_1_1641323676/checkpoints",
    #     "iterations": [499],
    # },
    "experiment": {
        "seed": 4,
        "steps": 2200,
        "iterations": 1000,
        "redundancy": 5,
        "env": "CartPole-v0",
    },
    "agent": {
        "class_": dqv_max.DQVMaxAgent,
        "net_sync_freq": 1e4,
        "min_replay_history": 5e4,
    },
    "logs": {
        # "path": "/home/xqz-u/uni/dopamine/resources/data/aim_mul_runs_new",
        "path": "/home/xqz-u/uni/thesis/resources/data/aim_mul_runs_new",
        "summary_writing_freq": 500,
    },
}


def make_agent(config: dict, env: gym.Env):
    observation_shape = env.observation_space.shape + (1,)
    return config["agent"]["class_"](
        config,
        num_actions=env.action_space.n,
        observation_shape=observation_shape,
        observation_dtype=env.observation_space.dtype,
    )


env = gym.make("CartPole-v0")
aim_conf = jax.tree_map(lambda l: str(l) if callable(l) else l, conf)
for i in range(conf["experiment"]["redundancy"]):
    run_log = aim.Run(repo=conf["logs"]["path"], experiment=f"dqv_max_cartpole_{i}")
    run_log["hparams"] = aim_conf
    ag = make_agent(conf, env)
    start = time.time()
    sr.run_exp(
        conf["experiment"]["iterations"],
        conf["experiment"]["steps"],
        ag,
        env,
        conf["logs"]["summary_writing_freq"],
        run_log,
    )
    end = time.time() - start
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} time: {end}s")
