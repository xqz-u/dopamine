import functools as ft
import itertools as it
import os
from concurrent import futures

from thesis import utils as u
from thesis.offline.replay_memory import offline_circular_replay_buffer as mem

BASE_DIR = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine"
THESIS = os.path.join(BASE_DIR, "thesis")
checkpoints_dir = os.path.join(THESIS, "tests/checkpoints")

obs_shape = (4, 1)
stack_sz = 1
rep_cap = 5000
batch_sz = 128


def print_merged_buffers_attrs(buf):
    for attr in ["add_count", "_replay_capacity", "invalid_range"]:
        print(getattr(buf, attr))


@u.timer
def load_buffers_serially(self, checkpoint_dir: str, iterations: list):
    for suffix in iterations:
        loader = self._load_buffer(checkpoint_dir, suffix)
        mem._merge_replay_buffers(self, loader)


@u.timer
def load_buffers_fold(self, checkpoint_dir: str, iterations: list):
    ft.reduce(
        mem._merge_replay_buffers,
        [self._load_buffer(checkpoint_dir, i) for i in iterations],
        self,
    )


@u.timer
def load_buffers_io_sub(self, checkpoint_dir: str, iterations: list):
    with futures.ThreadPoolExecutor(max_workers=4) as thread_pool:
        buffers = [
            thread_pool.submit(self._load_buffer, fold, i)
            # for fold, i in zip([checkpoint_dir] * len(iterations), iterations)
            for fold, i in zip(it.cycle([checkpoint_dir]), iterations)
        ]
    ft.reduce(mem._merge_replay_buffers, [b.result() for b in buffers], self)


@u.timer
def load_buffers_cpu(self, checkpoint_dir: str, iterations: list):
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        buffers = pool.map(
            self._load_buffer, [checkpoint_dir] * len(iterations), iterations
        )
    ft.reduce(mem._merge_replay_buffers, buffers, self)


@u.timer
def load_buffers_io(self, checkpoint_dir: str, iterations: list):
    with futures.ThreadPoolExecutor(max_workers=4) as thread_pool:
        buffers = thread_pool.map(
            self._load_buffer, [checkpoint_dir] * len(iterations), iterations
        )
    ft.reduce(mem._merge_replay_buffers, buffers, self)


def test_buffers_load_consistently():
    def test_one_load(m, iters):
        buf = mem.OfflineOutOfGraphReplayBuffer(obs_shape, stack_sz, batch_sz)
        getattr(buf, m)(checkpoints_dir, iters)
        print_merged_buffers_attrs(buf)

    test_one_load("load_single_buffer", "48")
    test_one_load("load_buffers", ["48"])
    test_one_load("load_buffers", ["49"])
    test_one_load("load_buffers", ["48", "49"])


# TODO do multiple timing tests to get average performance, do this once there
# are more training checkpoints to load
def test_loading_speeds():
    def test_one_method(m):
        buf = mem.OfflineOutOfGraphReplayBuffer(obs_shape, stack_sz, batch_sz)
        getattr(buf, m)(checkpoints_dir, ckpts)
        print_merged_buffers_attrs(buf)

    for method in [
        load_buffers_serially,
        load_buffers_fold,
        load_buffers_cpu,
        load_buffers_io,
        load_buffers_io_sub,
    ]:
        setattr(mem.OfflineOutOfGraphReplayBuffer, method.__name__, method)
    ckpts = [str(n) for n in range(46, 50)]
    test_one_method("load_buffers_serially")
    test_one_method("load_buffers_fold")
    test_one_method("load_buffers_cpu")
    test_one_method("load_buffers_io")
    test_one_method("load_buffers_io_sub")


test_buffers_load_consistently()
test_loading_speeds()

# buf = mem.OfflineOutOfGraphReplayBuffer(obs_shape, stack_sz, batch_sz)
# buf.load_single_buffer(checkpoints_dir, "49")
# buf.sample_transition_batch()
