import time

import aim
import gym
import jax
import optax
from dopamine.discrete_domains import gym_lib
from dopamine.jax import losses
from thesis import exploration
from thesis.agents import dqv_max
from thesis.runner import reporters
from thesis.tests import simple_runner as sr

# NOTE loss needs to be partialled right now if u want to pass args
conf = {
    "nets": {
        "qnet": {
            "model": {
                # "call_": ...,
                "hiddens": (512, 512),
            },
            "optim": {
                # "call_": ...,
                "learning_rate": 0.001
            },
            "loss": losses.huber_loss,
        },
        "vnet": {
            "model": {"hiddens": (512, 512)},
            "optim": {"learning_rate": 0.001},
            "loss": losses.huber_loss,
        },
    },
    # "exploration": {"call_": exploration.egreedy},
    "exploration": {},
    "agent": {
        "call_": dqv_max.DQVMaxAgent,
        "net_sync_freq": 1e4,
        "min_replay_history": 5e4,
        "observation_shape": (4, 1),
    },
    "env": {
        # "call_": gym_lib.create_gym_env,
        "environment_name": "MountainCar",
        "version": "v0",
    },
    "memory": {
        # "call_": outofgraph...,
        "stack_size": 1
    },
    # "logs": {
    #     # "path": "/home/xqz-u/uni/dopamine/resources/data/aim_mul_runs_new",
    #     "path": "/home/xqz-u/uni/thesis/resources/data/aim_mul_runs_new",
    #     "summary_writing_freq": 500,
    "runner": {
        "base_dir": "...",
        # "schedule": "continuous_train_and_eval",
        # "resume": False,
        "experiment": {
            # "seed": 4,
            "steps": 2200,
            "iterations": 1000,
            "redundancy": 5,
        },
        "reporters": [
            {
                "call_": reporters.AimReporter,
                "repo_path": "/home/xqz-u/uni/dopamine/resources/data/aim_mul_runs_new",
            }
        ],
    },
}


make_network = lambda name: {
    f"{name}": {
        "model": {"hiddens": (512, 512)},
        "optim": {
            "class_": optax.adam,
            "learning_rate": 0.001,
            "eps": 3.125e-4,
        },
        "loss": losses.huber_loss,
    }
}


def make_agent(config: dict, env: gym.Env):
    observation_shape = env.observation_space.shape + (1,)
    return config["agent"]["class_"](
        config,
        num_actions=env.action_space.n,
        observation_shape=observation_shape,
        observation_dtype=env.observation_space.dtype,
    )


def run(conf: dict, aim_exp: str):
    env = gym.make(conf["experiment"]["env"])
    aim_conf = jax.tree_map(lambda l: str(l) if callable(l) else l, conf)
    for i in range(conf["experiment"]["redundancy"]):
        run_log = aim.Run(repo=conf["logs"]["path"], experiment=f"{aim_exp}_{i}")
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


from pprint import pprint


def run_mountaincar():
    global conf
    c = conf.copy()
    c["experiment"]["env"] = "MountainCar-v0"
    c.pop("qnet", "vnet")
    qnet_adam = make_network("qnet")
    vnet_adam = make_network("vnet")
    models = qnet_adam | vnet_adam
    conf_mountaincar = {**c, "nets": models}
    pprint(conf_mountaincar)
    run(conf_mountaincar, "dqv_max_mountaincar")


conf_cartpole = {**conf, "env": "CartPole-v0"}

# run(conf_cartpole, "dqv_max_cartpole")

# run_mountaincar()


# off_mem = {
#     "memory": {
#         "class_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
#         "stack_size": 1,
#         "checkpoint_dir": "/home/xqz-u/uni/dopamine/resources/data/JaxDQNAgent_CartPole-v0_ClassicControlDQNNetwork_ref_1_1641323676/checkpoints",
#         "iterations": [499],
#     }
# }
