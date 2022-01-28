import os
from typing import Tuple

import tensorflow as tf
from dopamine.discrete_domains.checkpointer import Checkpointer
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Checkpointer


def get_latest_ckpt_number(base_directory: str) -> Tuple[int]:
    def extract_iteration(x: str):
        return tuple(map(int, x[x.rfind(".") + 1 :].split("_")))

    try:
        checkpoint_files = tf.io.gfile.glob(
            os.path.join(base_directory, "sentinel_checkpoint_complete.*")
        )
    except tf.errors.NotFoundError:
        return -1
    try:
        return max(extract_iteration(x) for x in checkpoint_files)
    except ValueError:
        return -1


def checkpoint_gen_filename(self, prefix: str, iteration: int) -> str:
    return os.path.join(self._base_directory, f"{prefix}.{self.redundancy}_{iteration}")


def _save_checkpoint(self, redundancy: int, iteration: int, data: dict):
    self.redundancy = redundancy
    return self.save_checkpoint(iteration, data)


def _load_checkpoint(self, redundancy: int, iteration: int) -> dict:
    self.redundancy = redundancy
    return self.load_checkpoint(iteration)


Checkpointer._generate_filename = checkpoint_gen_filename
Checkpointer._save_checkpoint = _save_checkpoint
Checkpointer._load_checkpoint = _load_checkpoint


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OutOfGraphReplayBuffer


def memory_gen_filename(self, ckpt_dir: str, name: str, suffix: str) -> str:
    return os.path.join(ckpt_dir, f"{name}_ckpt.{self.redundancy}_{suffix}.gz")


def _save(self, ckpt_dir: str, redundancy: int, iteration: int):
    self.redundancy = redundancy
    return self.save(ckpt_dir, iteration)


def _load(self, ckpt_dir: str, redundancy: int, iteration: int):
    self.redundancy = redundancy
    return self.load(ckpt_dir, iteration)


OutOfGraphReplayBuffer._generate_filename = memory_gen_filename
OutOfGraphReplayBuffer._save = _save
OutOfGraphReplayBuffer._load = _load
