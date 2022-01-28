import os
import pickle
from typing import Tuple

import attr
import tensorflow as tf
from thesis.utils import ConsoleLogger

CHECKPOINT_DURATION = 4

_logger = ConsoleLogger(name=__name__)


# NOTE this is not really useful for me, I am using the reporters
# representation which basically has the same aim. One of the two
# is redundant
@attr.s(auto_attribs=True)
class Logger:
    logging_dir: str = attr.ib(validator=attr.validators.instance_of(str))
    data: dict = attr.ib(factory=dict)
    enabled: bool = attr.ib(init=False, default=False)

    def __attrs_post_init__(self):
        if not self.logging_dir:
            _logger.info("Logging directory not specified, will not log.")
            return
        self.logging_dir = os.path.join(self.logging_dir, "logs")
        try:
            tf.io.gfile.makedirs(self.logging_dir)
        except tf.errors.PermissionDeniedError:
            pass
        # not raising an exception so that if you want to run an
        # experiment w/out logging, then no need to change the runner
        if not tf.io.gfile.exists(self.logging_dir):
            _logger.warning(
                f"Could not create directory {self.logging_dir}, logging will be disabled."
            )
            return
        self.enabled = True

    # TODO if keeping this, properly implement a dict interface
    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if self.enabled:
            self.data[key] = value

    def gen_filename(self, redundancy: int, iteration: int) -> str:
        fname = f"log_{redundancy}_{iteration}"
        return os.path.join(self.logging_dir, fname)

    def log_to_file(self, redundancy: int, iteration: int):
        if not self.enabled:
            _logger.warning("Logging is disabled.")
            return
        log_file = self.gen_filename(redundancy, iteration)
        with tf.io.gfile.GFile(log_file, "w") as fout:
            pickle.dump(self.data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        # After writing a checkpoint file, we garbage collect the log file
        # that is CHECKPOINT_DURATION versions old.
        stale_iteration = iteration - CHECKPOINT_DURATION
        if stale_iteration < 0:
            return
        stale_file = self.gen_filename(redundancy, stale_iteration)
        try:
            tf.io.gfile.remove(stale_file)
        except tf.errors.NotFoundError:
            pass


@attr.s(auto_attribs=True)
class Checkpointer:
    ckpt_dir: str = attr.ib(validator=attr.validators.instance_of(str))
    ckpt_frequency: int = 1
    ckpt_file_prefix: str = attr.ib(init=False, default="ckpt")
    sentinel_file_prefix: str = attr.ib(
        init=False, default="sentinel_checkpoint_complete"
    )

    @ckpt_dir.validator
    def validate_ckpt_dir(self, attribute, value: str):
        if not value:
            raise ValueError("No path provided to Checkpointer.")
        value = os.path.join(value, "checkpoints")
        try:
            tf.io.gfile.makedirs(value)
        except tf.errors.PermissionDeniedError:
            # We catch the PermissionDeniedError and issue a more useful exception.
            raise ValueError(f"Unable to create checkpoint path: {value}.")

    def __attrs_post_init__(self):
        self.ckpt_dir = os.path.join(self.ckpt_dir, "checkpoints")

    def gen_filename(self, prefix: str, redundancy: int, iteration: int) -> str:
        fname = f"{prefix}.{redundancy}_{iteration}"
        return os.path.join(self.ckpt_dir, fname)

    def save_checkpoint(self, redundancy: int, iteration: int, data: dict):
        if iteration % self.ckpt_frequency != 0:
            return
        with tf.io.gfile.GFile(
            self.gen_filename(self.ckpt_file_prefix, redundancy, iteration),
            "w",
        ) as fout:
            pickle.dump(data, fout)
        with tf.io.gfile.GFile(
            self.gen_filename(self.sentinel_file_prefix, redundancy, iteration),
            "wb",
        ) as fout:
            fout.write("done")
        self.clean_up_old_checkpoints(redundancy, iteration)

    def clean_up_old_checkpoints(self, redundancy: int, iteration: int):
        # After writing a the checkpoint and sentinel file, we garbage
        # collect files that are CHECKPOINT_DURATION *
        # self._ckpt_frequency versions old.
        stale_iteration = iteration - (self.ckpt_frequency * CHECKPOINT_DURATION)
        if stale_iteration < 0:
            return
        stale_file = self.gen_filename(
            self.ckpt_file_prefix, redundancy, stale_iteration
        )
        stale_sentinel = self.gen_filename(
            self.sentinel_file_prefix, redundancy, stale_iteration
        )
        try:
            tf.io.gfile.remove(stale_file)
            tf.io.gfile.remove(stale_sentinel)
        except tf.errors.NotFoundError:
            # Ignore if file not found.
            _logger.info("Unable to remove %s or %s.", stale_file, stale_sentinel)

    def load_checkpoint(self, redundancy: int, iteration: int) -> dict:
        ckpt_file = self.gen_filename(self.ckpt_file_prefix, redundancy, iteration)
        if not tf.io.gfile.exists(ckpt_file):
            return None
        with tf.io.gfile.GFile(ckpt_file, "rb") as fin:
            return pickle.load(fin)


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
