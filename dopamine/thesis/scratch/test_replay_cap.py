import numpy as np
from thesis import config


class Holder:
    cap: tuple
    store: np.ndarray

    def __init__(self, shape=(2,)):
        self.cap = shape
        self.store = np.empty(shape)

    def save(self, where):
        with open(where, "wb") as fd:
            np.save(fd, self.store, allow_pickle=False)

    def load(self, where):
        with open(where, "rb") as fd:
            self.store = np.load(fd, allow_pickle=False)


import os

p = f"{config.scratch_data_dir}/test_rep_cap"
os.makedirs(p, exist_ok=True)
x = Holder((4,))
for i in range(4):
    x.store[i] = i

x.save(p)

y = Holder()
y.load(p)


from dopamine.replay_memory import circular_replay_buffer

mem = circular_replay_buffer.OutOfGraphReplayBuffer(
    observation_shape=(4, 1), stack_size=1, replay_capacity=100
)

mem.save(p, 0)


mem_new = circular_replay_buffer.OutOfGraphReplayBuffer(
    observation_shape=(4, 1), stack_size=1
)
mem_new.load(p, 0)
