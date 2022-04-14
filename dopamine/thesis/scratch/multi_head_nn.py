from dopamine.discrete_domains import atari_lib
from jax import random as jrand
from thesis import custom_pytrees, networks, utils
from thesis.agents import agent_utils

# TODO:
# - design a multi-head networks blueprint which can split the final
#   layer in N dense layers, using the same body of parameters for the
#   previous layers

rng = custom_pytrees.PRNGKeyWrap()

sample_shape = (
    (2,) + atari_lib.NATURE_DQN_OBSERVATION_SHAPE + (atari_lib.NATURE_DQN_STACK_SIZE,)
)

example_in, example_state = jrand.randint(rng.key, sample_shape, 0, 256)

net, params, rest = agent_utils.build_net(2, example_in, rng, networks.NatureDQNNetwork)

utils.jax_container_shapes(params)

action_qvals = net.apply(params, example_state)


# import os

# from thesis import config
# from thesis.memory import offline_memory

# pong_buffers_path = os.path.join(config.data_dir, "Pong/1/replay_logs")

# off_mem = offline_memory.OfflineOutOfGraphReplayBuffer(
#     pong_buffers_path,
#     _buffers_iterations=[1],
#     replay_capacity=int(1e6),
#     batch_size=4,
#     observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
#     stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
# )
# off_mem.load_single_buffer(1)
# atari_records = agent_utils.sample_replay_buffer(off_mem)
# atari_states = atari_records["state"]
# example_in, example_state, *atari_states = atari_states
# del off_mem  # consumes 7G of memory this mf
