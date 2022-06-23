import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from attrs import define, field

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


def default_method(self, arg: int):
    self.a = arg
    print("default method")


def another_method(self, arg: int):
    self.a = arg + 5
    print("another method")


import types


@define
class Pippo:
    a: int = field(init=False)
    method: callable = field(default=default_method)

    def __attrs_post_init__(self):
        print(self.method)
        self.method = self.method.__get__(self)
        # self.method = types.MethodType(self.method, self)

    def caller(self, arg):
        self.method(arg)
        print(self.a)


x = Pippo()
x.caller(5)

y = Pippo(method=another_method)
y.caller(5)
