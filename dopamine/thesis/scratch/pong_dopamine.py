import os

import gin
from dopamine.discrete_domains import run_experiment
from thesis import config

pong_gin_conf = os.path.join(
    config.dopamine_dir, "jax/agents/dqn/configs/dqn_profiling.gin"
)
data_path = os.path.join(config.scratch_dir, "Pong", "dqn")
gin.parse_config_file(pong_gin_conf)
run = run_experiment.create_runner(data_path)


import random

import gym

render_mode = "rgb_array"
env = gym.make("ALE/Pong-v5", render_mode=render_mode)
init_obs = env.reset()
for i in range(1000):
    action = random.randint(0, 5)
    obs, reward, done, _ = env.step(action)
    if done:
        print(f"DEAD at {i} step!")
        break
env.close()
