from typing import List

from dopamine.replay_memory.prioritized_replay_buffer import (
    OutOfGraphPrioritizedReplayBuffer,
)
from thesis import utils
from thesis.memory.offline_memory import OfflineOutOfGraphReplayBuffer


# TODO merge multiple buffers!
# https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
# for a solution to inherit a parent's property setter
class PrioritizedOfflineOutOfGraphReplayBuffer(
    OfflineOutOfGraphReplayBuffer, OutOfGraphPrioritizedReplayBuffer
):
    _buffers_dir: str
    _parent_necessary_attributes: List[str]

    def __init__(
        self, _buffers_dir: str, _buffers_iterations: List[int] = None, **kwargs
    ):
        super().__init__(
            _buffers_dir=_buffers_dir, _buffers_iterations=_buffers_iterations, **kwargs
        )

    @property
    def parent_necessary_attributes(self) -> List[str]:
        return self._parent_necessary_attributes + ["sum_tree"]

    # override
    def _load_buffer(self, suffix: int) -> OutOfGraphPrioritizedReplayBuffer:
        buff = super()._load_buffer(suffix)
        prio_buff = OutOfGraphPrioritizedReplayBuffer(**self._kwargs)
        for trajectory in zip(*buff._store.values()):
            prio_buff.add(
                *trajectory, priority=prio_buff.sum_tree.max_recorded_priority
            )
        return prio_buff

    # override
    def _merge_two_buffers(self, other: OutOfGraphPrioritizedReplayBuffer):
        pass

    # override (temporary!)
    def load_buffers(self, _: bool, iterations: List[int] = None):
        if iterations is None:
            iterations = utils.list_all_ckpt_iterations(self._buffers_dir)[0]
            self._console.debug(f"TEMP iterations was none, fix to {iterations}")
        elif len(iterations) > 1:
            iterations = iterations[0]
            self._console.debug(
                f"TEMP cannot load multiple buffers rn, loading only the first given one: {iterations}"
            )
        self.load_single_buffer(iterations)
