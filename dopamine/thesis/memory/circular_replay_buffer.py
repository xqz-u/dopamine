import logging

import numpy as np
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer

# PATCHING
# patch OutOfGraphReplayBuffer to correctly save all experiences
# encountered during training. since the memory is a circular buffer,
# it needs to be checkpointed every time it is full or transitions are
# overridden. moreoever, it needs to be flushed when it is about to be
# gc'ed: if `OutOfGraphReplayBuffer.replay_capacity` % #total number
# of trajectories that will be added != 0, some trajectories remain
# pending; in that case, write one last checkpoint containing only the
# remaining trajectories (`save_buf_subset`)


logger = logging.getLogger(__name__)


def save_buff_subset(self, suffix: int):
    last_trans_idx = self.cursor()
    logger.debug(
        f"Save {last_trans_idx} pending transitions out of {self._replay_capacity}"
    )
    # `Dopamine.OutOfGraphReplayBuffer` saves element from self._store,
    # so swap it out temporarily to save only the trajectories
    # encountered so far - ahead there are old ones to be overwritten
    latest_trans = {k: self._store[k][:last_trans_idx] for k in self._store.keys()}
    og_buff = self._store
    og_add_count = self.add_count
    # adapt checkpointable attributes to the saved memory subset
    self._store = latest_trans
    self.add_count = np.array(last_trans_idx)
    self.save(self._full_experience_path, suffix)
    self._store = og_buff
    self.add_count = og_add_count


def add_with_full_exp(self, *args, **kwargs):
    self._og_add(*args, **kwargs)
    if self.cursor() == 0:
        # full circular buffer, write checkpoint
        logger.debug(
            f"Saving memory at {self._full_experience_path}_{self._n_snapshots}"
        )
        self.save(self._full_experience_path, self._n_snapshots)
        self._n_snapshots += 1


def finalize_full_experience(self):
    if self._tot_budget < self._replay_capacity:
        # the buffer was never full so never saved, dump it
        logger.info(f"Dumping {self.add_count} trajectories at once")
        self.save(self._full_experience_path, self._n_snapshots)
    elif self.add_count % self._tot_budget:
        # there are some residual transitions to be saved
        self.save_buff_subset(self._n_snapshots)


def full_experience_initializer(self, checkpoint_dir: str, steps: int, iterations: int):
    # the total number of trajectories that will be added during training
    self._tot_budget = steps * iterations
    # counter for the number of dumped checkpoints
    self._n_snapshots = 0
    # NOTE this should be the same as `runner.base.checkpoint_dir`
    self._full_experience_path = checkpoint_dir
    # runtime monkeypatching
    OutOfGraphReplayBuffer.add = add_with_full_exp


OutOfGraphReplayBuffer._og_add = OutOfGraphReplayBuffer.add
for fn in [
    save_buff_subset,
    full_experience_initializer,
    add_with_full_exp,
    finalize_full_experience,
]:
    setattr(OutOfGraphReplayBuffer, fn.__name__, fn)
