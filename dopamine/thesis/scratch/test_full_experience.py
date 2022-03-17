import os

import gym
import numpy as np
from thesis import config, constants, offline_circular_replay_buffer, utils
from thesis.runner import runner
from thesis.scratch import test_mongo_reporter

cartpole = gym.make("CartPole-v0")
cartpole_dict = constants.env_info("CartPole", "v0")
memory_args = {
    "observation_shape": cartpole_dict["observation_shape"],
    "stack_size": 1,
    "observation_dtype": cartpole.observation_space.dtype,
}

conf = test_mongo_reporter.make_config("test_exp_recorder")
conf["runner"]["exp_recorder"] = True
conf["runner"]["experiment"]["redundancy"] = 2
conf["memory"]["replay_capacity"] = 1500
# utils.data_dir_from_conf(conf["experiment_name"], conf)
# run = runner.create_runner(conf)
# run.run_experiment_with_redundancy()


experiences_path = os.path.join(
    config.data_dir,
    "CartPole-v0/DQVMaxAgent/test_exp_recorder/checkpoints/redundancy_0/full_experience",
)


off_memory = offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer(
    **memory_args, checkpoint_dir=experiences_path, iterations=list(range(0, 5))
)
# off_memory.load_buffers()
# for x in range(0, 5):
#     off_memory._load_buffer(x)
#     print("--------------------")
# off_memory.load_single_buffer(n)

# x = np.load(os.path.join(experiences_path, "add_count_ckpt.4.gz"), allow_pickle=True)
