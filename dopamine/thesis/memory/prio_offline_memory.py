from typing import List

from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from dopamine.replay_memory.prioritized_replay_buffer import (
    OutOfGraphPrioritizedReplayBuffer,
)
from thesis import utils
from thesis.memory.offline_memory import OfflineOutOfGraphReplayBuffer

dopamine_memory_defaults = utils.callable_defaults(OutOfGraphReplayBuffer.__init__)
prio_dopamine_missing_defaults = ["replay_capacity", "batch_size"]


# NOTE
# https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
# provides some solution to inherit a parent's property setter, but it
# is very convoluted, so raise default AttributeError on property
# setter here...
class PrioritizedOfflineOutOfGraphReplayBuffer(
    OfflineOutOfGraphReplayBuffer, OutOfGraphPrioritizedReplayBuffer
):
    _parent_necessary_attributes: List[str]

    def __init__(self, *args, **kwargs):
        # NOTE prioritized_replay_buffer wants replay_capacity and
        # batch_size too. Add the base defaults if not present, else
        # remember to pass these arguments as keywords!
        for key in prio_dopamine_missing_defaults:
            kwargs[key] = kwargs.get(key, dopamine_memory_defaults[key])
        super().__init__(*args, **kwargs)

    @property
    def parent_necessary_attributes(self) -> List[str]:
        return self._parent_necessary_attributes + ["sum_tree"]

    # override
    def _load_buffer(self, suffix: int):
        ...

    # override
    def _merge_two_buffers(self, other: OutOfGraphPrioritizedReplayBuffer):
        ...
