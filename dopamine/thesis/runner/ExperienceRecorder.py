from typing import Union

import attr
from dopamine import replay_memory


@attr.s(auto_attribs=True)
class ExperienceRecorder:
    memory: Union[
        replay_memory.circular_replay_buffer.OutOfGraphReplayBuffer,
        replay_memory.prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
    ]
    path: str
    last_iteration: str = attr.ib(init=False, default="0")

    # snapshot the whole memory once the buffer is about to get
    # overwritten
    def snapshot_experience(self, iteration: int):
        if self.memory.cursor() == 0:
            self.memory.save(self.path, f"{self.last_iteration}-{iteration}")
            self.last_iteration = iteration


def add_hook(func: callable, hook: callable):
    ...
