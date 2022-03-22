import functools as ft
import math
import multiprocessing as mp
import os
from concurrent import futures

import numpy as np
from dopamine.replay_memory import circular_replay_buffer

from thesis import utils


def _merge_replay_buffers(self, other):
    self.invalid_range = np.concatenate(
        [self.invalid_range, self._replay_capacity + other.invalid_range]
    )
    self.add_count += other.add_count
    self._replay_capacity += other._replay_capacity
    self._store = dict(
        zip(
            self._store.keys(),
            [
                np.concatenate(arrays)
                for arrays in zip(self._store.values(), other._store.values())
            ],
        )
    )
    return self


class OfflineOutOfGraphReplayBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):
    def __init__(
        self,
        observation_shape: tuple,
        stack_size: int,
        _buffers_root_dir: str,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(
            observation_shape,
            stack_size,
            batch_size,
            replay_capacity=math.inf,
            **kwargs,
        )
        self.invalid_range = np.array([], dtype=np.int32)
        self._kwargs = kwargs
        self._buffers_root_dir = _buffers_root_dir

    # NOTE override of parent's method, in parent's __init__
    def _create_storage(self):
        self._replay_capacity = 0
        super()._create_storage()

    def _load_buffer(self, buffer_dir: str, suffix: int):
        loader = self.__class__(
            self._observation_shape,
            self._stack_size,
            self._buffers_root_dir,
            batch_size=self._batch_size,
            **self._kwargs,
        )
        loader.load(buffer_dir, suffix)
        assert utils.all_equal([v.shape[0] for v in loader._store.values()])
        loader._replay_capacity = loader._store["observation"].shape[0]
        return loader

    def _validate_capacity(self):
        if self._replay_capacity < self._update_horizon + self._stack_size:
            raise ValueError(
                "There is not enough capacity to cover "
                "update_horizon and stack_size."
            )
        if self._replay_capacity < self._batch_size:
            raise ValueError(
                f"Not enough trajectories are available for sampling: capacity {self._replay_capacity}, batch size: {self._batch_size}"
            )

    def load_single_buffer(self, buffer_dir: str, iteration: int):
        other = self._load_buffer(buffer_dir, iteration)
        for attr in ["add_count", "_replay_capacity", "invalid_range", "_store"]:
            setattr(self, attr, getattr(other, attr))
        self._validate_capacity()

    def load_buffers(self, suffix: str, workers: int = mp.cpu_count() - 1):
        buffers_dir = os.path.join(self._buffers_root_dir, suffix)
        dir_iterations = utils.list_all_ckpt_iterations(buffers_dir)
        with futures.ThreadPoolExecutor(max_workers=workers) as thread_pool:
            buffers = [
                thread_pool.submit(self._load_buffer, buffers_dir, i)
                for i in dir_iterations
            ]
        ft.reduce(_merge_replay_buffers, [b.result() for b in buffers], self)
        self._validate_capacity()
        print(f"loaded all buffers from {buffers_dir}")
