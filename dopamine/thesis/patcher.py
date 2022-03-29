import gzip
import os
import pickle

import numpy as np
import tensorflow as tf
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer


def save_no_garbage(self, checkpoint_dir, iteration_number):
    if not tf.io.gfile.exists(checkpoint_dir):
        return
    checkpointable_elements = self._return_checkpointable_elements()
    for attr in checkpointable_elements:
        filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
        with tf.io.gfile.GFile(filename, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
                if attr.startswith(circular_replay_buffer.STORE_FILENAME_PREFIX):
                    array_name = attr[
                        len(circular_replay_buffer.STORE_FILENAME_PREFIX) :
                    ]
                    np.save(outfile, self._store[array_name], allow_pickle=False)
                # Some numpy arrays might not be part of storage
                elif isinstance(self.__dict__[attr], np.ndarray):
                    np.save(outfile, self.__dict__[attr], allow_pickle=False)
                else:
                    pickle.dump(self.__dict__[attr], outfile)


# NOTE would be better to have a factory classmethod to create an
# instance of memory given a a buffer, but this in quicker for now
def save_buff_subset(self, save_path: str, suffix: int):
    last_trans_idx = self.cursor() - 1
    print(f"save {last_trans_idx} pending transitions out of {self._replay_capacity}")
    latest_trans = {k: self._store[k][:last_trans_idx] for k in self._store.keys()}
    og_buff = self._store
    og_add_count = self.add_count
    og_invalid_range = self.invalid_range
    # adapt checkpointable attributes to saved memory subset
    self._store = latest_trans
    self.add_count = np.array(last_trans_idx)
    self.invalid_range = circular_replay_buffer.invalid_range(
        self.cursor(), self._replay_capacity, self._stack_size, self._update_horizon
    )
    self.save_no_garbage(save_path, suffix)
    self._store = og_buff
    self.add_count = og_add_count
    self.og_invalid_range = og_invalid_range


def add_with_full_exp(self, *args, **kwargs):
    self._og_add(*args, **kwargs)
    if self.cursor() == 0:
        print(f"saving at {self._full_experience_path}_{self._n_snapshots}")
        self.save_no_garbage(self._full_experience_path, self._n_snapshots)
        self._n_snapshots += 1


def finalize_full_experience(self):
    if self._replay_capacity % self._tot_budget == 0:
        # no residual transitions, already saved by add_with_full_exp
        return
    self.save_buff_subset(self._full_experience_path, self._n_snapshots)
    print("Saved residual experiences")


def full_experience_initializer(
    self, full_experience_path: str, steps: int, iterations: int
):
    self._tot_budget = steps * iterations
    self._n_snapshots = 0
    # NOTE this is the same as self._base_directory
    self._full_experience_path = full_experience_path
    os.makedirs(self._full_experience_path, exist_ok=True)
    OutOfGraphReplayBuffer.add = add_with_full_exp


OutOfGraphReplayBuffer._og_add = OutOfGraphReplayBuffer.add
for fn in [
    save_buff_subset,
    save_no_garbage,
    full_experience_initializer,
    add_with_full_exp,
    finalize_full_experience,
]:
    setattr(OutOfGraphReplayBuffer, fn.__name__, fn)
