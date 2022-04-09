from concurrent import futures
from typing import List

import numpy as np
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from thesis import utils


# NOTE the idea behind creating a new class for offline memory was
# originally that it could share some core functionality with its
# prioritized counterpart; if this is not true, a class can be
# avoided: the same logic to merge OutOfGraphReplayBuffer(s) can be
# implemented with a function which just returns the latter...
# NOTE the parameters passed as kwargs won't be detected when the
# memory is instantiated (see thesis.agents.Agent:68); the ones
# declared explicitly in this class' init are those which make sense
# here
class OfflineOutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
    _buffers_dir: str
    _console: utils.ConsoleLogger
    _parent_necessary_attributes: List[str] = [
        "add_count",
        "_replay_capacity",
        "invalid_range",
        "_store",
    ]

    def __init__(
        self, _buffers_dir: str, _buffers_iterations: List[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._kwargs = kwargs
        self._buffers_dir = _buffers_dir
        self._console = utils.ConsoleLogger(name=__name__)
        self.load_buffers(iterations=_buffers_iterations)

    @property
    def parent_necessary_attributes(self) -> List[str]:
        return self._parent_necessary_attributes

    @parent_necessary_attributes.setter
    def parent_necessary_attributes(self, _):
        raise Exception(
            f"Attribute _parent_necessary_attributes of {self.__class__} is frozen"
        )

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

    def _load_buffer(self, suffix: int) -> OutOfGraphReplayBuffer:
        buff = OutOfGraphReplayBuffer(**self._kwargs)
        buff.load(self._buffers_dir, suffix)
        buff._replay_capacity = buff._store["observation"].shape[0]
        return buff

    def _merge_buffers(self, bf: OutOfGraphReplayBuffer):
        self.add_count += bf.add_count
        self._replay_capacity += bf._replay_capacity
        self._store = dict(
            zip(
                self._store.keys(),
                [
                    np.concatenate(arrays)
                    for arrays in zip(self._store.values(), bf._store.values())
                ],
            )
        )

    def load_single_buffer(self, suffix: int):
        other = self._load_buffer(suffix)
        for attr in self.parent_necessary_attributes:
            setattr(self, attr, getattr(other, attr))
        self._validate_capacity()
        self._console.debug(
            f"Loaded buffer {self._buffers_dir}-{suffix} add_count: {self.add_count}"
        )

    # default behavior: load all iterations in a directory if not
    # specified otherwise
    def load_buffers(self, iterations: List[int] = None, workers: int = None):
        if iterations is None:
            iterations = utils.list_all_ckpt_iterations(self._buffers_dir)
            self._console.debug(f"Load all buffers in {self._buffers_dir}")
        with futures.ThreadPoolExecutor(max_workers=workers) as thread_pool:
            buffers = [thread_pool.submit(self._load_buffer, i) for i in iterations]
        for bf in [b.result() for b in buffers]:
            self._merge_buffers(bf)
        self._validate_capacity()
        self._console.debug(f"loaded buffers {iterations} from {self._buffers_dir}")
