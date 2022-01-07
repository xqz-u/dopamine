import time
from functools import partial

import gym
from aim import Run
from jax import numpy as jnp
from thesis import experiment_data as ed
from thesis.jax import optimizers
from thesis.jax.agents.dqv_family import dqv_max_agent
from thesis.tests import simple_runner as sr


def mse(x, y):
    return jnp.power(x - y, 2)


env = gym.make("CartPole-v0")
exp_data = ed.ExperimentData(
    create_optimizer_fn=partial(optimizers.adam_optimizer, 3.125e-4),
    loss_fn=mse,
    min_replay_history=5e4,
    target_update_period=1e4,
)

path = "/home/xqz-u/uni/dopamine/resources/data/aim_mul_runs_2"
for i in range(10):
    run_log = Run(repo=path, experiment=f"dqv_max_cartpole_{i}")
    run_log["hparams"] = {
        "optimizer": {"name": "adam", "lr": 3.125e-4},
        "net": {
            "V": {"spec": "dense-relu", "arch": [512, 512]},
            "Q": {"spec": "dense-relu", "arch": [512, 512]},
        },
        "loss": "mse",
        "steps": 2200,
        "hist_max_size": 1e6,
        "min_replay_hist": 5e4,
        "batch_size": 32,
        "target_upd_p": 1e4,
        "epochs": 1000,
        "eps": 0.01,
        "reps": 10,
        "env": "CartPole-v0",
    }
    ag = dqv_max_agent.JaxDQVMaxAgent((4, 1), 2, exp_data, summary_writer=run_log)
    start = time.time()
    sr.run_exp(1000, 2200, ag, env)
    end = time.time() - start
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} time: {end}s")
