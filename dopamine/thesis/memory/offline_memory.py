import functools as ft
import logging
from concurrent import futures
from typing import List

import gin
import numpy as np
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from thesis import utils

logger = logging.getLogger(__name__)


# important attributs used to accumulate the returned buffer:
# - add_count: the total number of calls to OutOfGraphReplayBuffer.add,
#   so including overwritten trajectories. used to decide where to add
#   new trajectory (although this does not occur in offline RL)
# - store: the final trajectories are the concatenation of individual
#   buffers
def merge_buffers(
    acc: OutOfGraphReplayBuffer, other: OutOfGraphReplayBuffer
) -> OutOfGraphReplayBuffer:
    acc.add_count += other.add_count
    acc._replay_capacity += other._replay_capacity
    acc._store = dict(
        zip(
            acc._store.keys(),
            [
                np.concatenate(arrays)
                for arrays in zip(acc._store.values(), other._store.values())
            ],
        )
    )
    return acc


def load_buffer(
    buffers_dir: str, iteration_suffix: int, **kwargs
) -> OutOfGraphReplayBuffer:
    buff = OutOfGraphReplayBuffer(**kwargs)
    buff.load(buffers_dir, iteration_suffix)
    # infer real replay capacity of dumped buffer - the loader one might
    # have been created with a big placeholder one
    buff._replay_capacity = buff._store["observation"].shape[0]
    return buff


@gin.configurable
def load_offline_buffers(
    buffers_dir: str, iterations: List[int] = None, parallel: bool = False, **kwargs
) -> OutOfGraphReplayBuffer:
    # load all memory checkpoints available in a folder if not
    # specified otherwise
    if iterations is None:
        iterations = utils.list_all_ckpt_iterations(buffers_dir)
        logger.info(f"Load all buffers in {buffers_dir}")
    first_iter, *rest_iter = iterations
    if not parallel:
        buffers_union = ft.reduce(
            lambda acc, i: merge_buffers(acc, load_buffer(buffers_dir, i, **kwargs)),
            rest_iter,
            load_buffer(buffers_dir, first_iter, **kwargs),
        )
    else:
        logger.info("Load buffers in parallel with a thread pool")
        with futures.ThreadPoolExecutor() as thread_pool:
            buffers = [
                thread_pool.submit(load_buffer, buffers_dir, i, **kwargs)
                for i in iterations
            ]
        buffers_union = ft.reduce(merge_buffers, (b.result() for b in buffers))
    logger.info(f"Loaded buffers {iterations} from {buffers_dir}")
    return buffers_union
