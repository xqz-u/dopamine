from collections import namedtuple
from concurrent import futures
from typing import List, Tuple

import numpy as np
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from dopamine.replay_memory.prioritized_replay_buffer import (
    OutOfGraphPrioritizedReplayBuffer,
)
from thesis import utils


# NOTE from
# https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/configs/dqn.gin
# it seems that the DQN replay buffers, the ones I'm really interested
# in, are all created with the default size of 1M. If this is not the
# case when running my tests/experiments, remember the replay_capacity
# used; not checkpointing it or there will be an error when trying to
# read buffers which did not save this information, i.e. the DQN RD.
# replay_capacity is used to get cursor(), it is also used for sampling
# transitions
class OfflineOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
    _buffers_dir: str
    _replay_buffers: List[OutOfGraphReplayBuffer]
    _console: utils.ConsoleLogger

    def __init__(self, _buffers_dir: str, *args, **kwargs):
        super().__init__(*args ** kwargs)
        self._buffers_dir = _buffers_dir
        self._replay_buffers = []
        self._args, self._kwargs = args, kwargs
        self._console = utils.ConsoleLogger(name=__name__)

    @property
    def n_buffers(self):
        return len(self._replay_buffers)

    def _load_buffer(self, suffix: int) -> OutOfGraphReplayBuffer:
        buff = OutOfGraphReplayBuffer(*self._args, **self._kwargs)
        buff.load(self._buffers_dir, suffix)
        return buff

    def load_single_buffer(self, suffix: int):
        buff = self._load_buffer(suffix)
        self._replay_buffers.append(buff)
        self._console.debug(
            f"Loaded buffer {self._buffers_dir}-{suffix} add_count: {buff.add_count}"
        )

    def load_buffers(self, workers: int = None):
        with futures.ThreadPoolExecutor(max_workers=workers) as thread_pool:
            buffers = [
                thread_pool.submit(self._load_buffer, i)
                for i in utils.list_all_ckpt_iterations(self._buffers_dir)
            ]
        self._replay_buffers = [b.result() for b in buffers]
        print(f"loaded {self.n_buffers} buffers from {self._buffers_dir}")

    # override
    def get_transition_elements(self, batch_size=None) -> namedtuple:
        return self._replay_buffers[0].get_transition_elements(batch_size)

    def sample_transition_batch(
        self, batch_size=None, indices=None
    ) -> Tuple[np.ndarray, ...]:
        buffer_index = np.random.randint(self._num_replay_buffers)
        self._console.debug("Sampling from buffer #{buffer_index}")
        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices
        )


# FIXME problem: with this approach, a buffer from wich to perform
# stratified sampling needs to be chosen first. How to choose it without
# changing the underlying distribution?
# An alternative could be to merge the replay buffers, as I was doing
# previously. Maybe I should focus on the ensemble DQVMax right now? It
# seems like some thinking is needed here...
# NOTE remember to pass replay_capacity and stack_size, they don't
# have defaults in OutOfGraphPrioritizedReplayBuffer
class PrioritizedOfflineOutOfGraphReplayBuffer(OfflineOutOfGraphReplayBuffer):
    _buffers_dir: str
    _replay_buffers: List[OutOfGraphPrioritizedReplayBuffer]
    _console: utils.ConsoleLogger

    def __init__(self, _buffers_dir: str, *args, **kwargs):
        super().__init__(_buffers_dir, *args, **kwargs)

    # override, takes care of loading a checkpoint produced by a
    # non-prioritized replay buffer
    def _load_buffer(self, suffix: int) -> OutOfGraphPrioritizedReplayBuffer:
        loader_buff = OutOfGraphReplayBuffer(*self._args, **self._kwargs)
        loader_buff.load(self._buffers_dir, suffix)
        prio_buff = OutOfGraphPrioritizedReplayBuffer(*self._args, **self._kwargs)
        for transition in zip(*loader_buff._store.values()):
            prio_buff.add(
                *transition, priority=prio_buff.sum_tree.max_recorded_priority
            )
        return prio_buff


# import os

# from dopamine.discrete_domains import atari_lib
# from thesis import config
# from thesis.agents import agent_utils

# pip = OutOfGraphReplayBuffer(
#     observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
#     stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
#     replay_capacity=100,
# )
# p = os.path.join(config.data_dir, "Pong/1/replay_logs")
# pip.load(p, 1)


# minibatch = agent_utils.sample_replay_buffer(pip, batch_size=5)

# # pip = OutOfGraphPrioritizedReplayBuffer(
# #     observation_shape=(4, 1), stack_size=1, batch_size=32, replay_capacity=int(1e6)
# # )
