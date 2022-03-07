import os

import gym
from dopamine.replay_memory import prioritized_replay_buffer
from thesis import config, constants, offline_circular_replay_buffer
from thesis.agents import agent_utils

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


per_memory = prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
    **memory_args, replay_capacity=off_memory._replay_capacity, batch_size=5
)
trans = [
    {k: v[i] for k, v in minibatch.items()} for i in range(len(minibatch["state"]))
]
per_trans = [
    {
        k: v.squeeze(axis=1) if k == "state" else v
        for k, v in t.items()
        if k in ["state", "action", "reward", "terminal"]
    }
    for t in trans
]

# NOTE this does not work: we need to give a priority too, in the
# paper it is the td-error. How do I compute it for offline
# trajectories?
for t in per_trans:
    per_memory.add(*list(t.values()))
