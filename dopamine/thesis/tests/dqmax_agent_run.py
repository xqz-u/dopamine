from functools import partial

import gym
from aim import Run
from jax import numpy as jnp
from thesis import experiment_data as ed
from thesis.jax import optimizers
from thesis.jax.agents.dqv_family import dqv_max_agent
from thesis.tests import simple_runner as sr

run_log = Run(run_hash="egg", repo="/home/xqz-u/uni/thesis/resources/data/aim_Run")
mse = lambda x, y: jnp.power(x - y, 2)
env = gym.make("CartPole-v0")
exp_data = ed.ExperimentData(
    create_optimizer_fn=partial(optimizers.adam_optimizer, 3.125e-4),
    loss_fn=mse,
    min_replay_history=5e4,
    target_update_period=1e4,
)
ag = dqv_max_agent.JaxDQVMaxAgent((4, 1), 2, exp_data, summary_writer=run_log)

for i in range(10):
    sr.run_exp(1000, 2200, ag, env)
