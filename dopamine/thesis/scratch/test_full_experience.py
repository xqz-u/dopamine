import os

import gym
from dopamine.replay_memory import prioritized_replay_buffer
from thesis import config, constants, offline_circular_replay_buffer, utils
from thesis.agents import agent_utils
from thesis.runner import runner
from thesis.scratch import test_mongo_reporter

cartpole = gym.make("CartPole-v0")
cartpole_dict = constants.env_info("CartPole", "v0")
memory_args = {
    "observation_shape": cartpole_dict["observation_shape"],
    "stack_size": 1,
    "observation_dtype": cartpole.observation_space.dtype,
}


experiences_path = os.path.join(
    config.data_dir, "CartPole-v0/JaxDQNAgent/online_train/checkpoints"
)
off_memory = offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer(
    **memory_args, checkpoint_dir=experiences_path, iterations=[499]
)
off_memory.load_buffers()

minibatch = agent_utils.sample_replay_buffer(off_memory, batch_size=5)


trans_one = {k: minibatch[k][0] for k in minibatch.keys()}
trans_add = {
    k: v for k, v in trans_one.items() if k in ["state", "action", "reward", "terminal"]
}
trans_add["state"] = trans_add["state"].squeeze(1)

conf = test_mongo_reporter.make_config("test_exp_recorder")
conf["runner"]["exp_recorder"] = True
conf["runner"]["experiment"]["redundancy"] = 3
conf["memory"]["replay_capacity"] = 1500
# utils.data_dir_from_conf(conf["experiment_name"], conf)
# run = runner.create_runner(conf)
# run.setup_experience_recorder()
# run.run_experiment_with_redundancy()
