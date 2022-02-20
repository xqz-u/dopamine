import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminal")
)


@dataclass
class ReplayMemory:
    capacity: int
    batch_size: int
    memory: deque = None

    def __post_init__(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self) -> Dict[str, torch.tensor]:
        """
        Sample `self.batch_size` `Transition`s, and return them as
        a batched dict.
        """
        return {
            k: torch.tensor(np.array(v))
            # k: torch.cat(v)
            for k, v in Transition(*zip(*random.sample(self.memory, self.batch_size)))
            ._asdict()
            .items()
        }

    def __len__(self):
        return len(self.memory)
