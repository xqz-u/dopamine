# import cProfile
import timeit

code = """
import gym
from dopamine.discrete_domains import atari_lib
from jax import random as jrand
from thesis import config, constants, custom_pytrees, networks, utils
from thesis.agents import agent_utils
from thesis.memory import offline_memory

env = gym.make("ALE/Pong-v5")
n_actions = env.action_space.n

rng = custom_pytrees.PRNGKeyWrap(44)
atari_shape = atari_lib.NATURE_DQN_OBSERVATION_SHAPE + (
    atari_lib.NATURE_DQN_STACK_SIZE,
)
example_inp = jrand.randint(next(rng), atari_shape, -1, 1)
net, params, _ = agent_utils.build_net(
    n_actions, example_inp, rng, networks.NatureDQNNetwork
)
print(utils.jax_container_shapes(params))

mem = offline_memory.OfflineOutOfGraphReplayBuffer(
    _buffers_dir=f"{config.data_dir}/Pong/1/replay_logs",
    _buffers_iterations=[10],
    load_parallel=False,
    observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
    stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
    **constants.default_memory_args,
)
minibatch = agent_utils.sample_replay_buffer(mem, batch_size=256)

# cProfile.run("agent_utils.batch_net_eval(net, params, minibatch['state'])")
"""

elapsed = timeit.timeit(
    stmt="agent_utils.batch_net_eval(net, params, minibatch['state'])",
    setup=code,
    number=10,
)
# stern: 2.5246042240032693
