import functools as ft
import os
import time
from itertools import groupby
from typing import Any, Dict, Tuple, Union

import gin
import tensorflow as tf
from aim import Run
from dopamine.replay_memory import circular_replay_buffer

from jax import numpy as jnp
from thesis import config
from thesis.offline.replay_memory import offline_circular_replay_buffer


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def timer(want_time=False):
    def decorator(fn):
        @ft.wraps(fn)
        def inner(*args, **kwargs) -> Union[Any, Tuple[Any, float]]:
            start = time.time()
            ret = fn(*args, **kwargs)
            end = time.time() - start
            if not want_time:
                print(f"`{fn.__name__}` exec time: {end}")
            return ret if not want_time else (ret, end)

        return inner

    # trick to keep writing @timer, @timer() and @timer(True)
    if callable(want_time):
        f = want_time
        want_time = False
        return decorator(f)
    return decorator


def dict_pop(d, k):
    return (d, d.pop(k))[0]


def mget(d, *keys):
    return [d[k] for k in keys]


# experiment_spec = [agent name, network name, environment name, *]
@gin.configurable
def make_unique_data_dir(experiment_spec: list, base_dir: str = None) -> str:
    agent, net, env_name, *args = experiment_spec
    if not base_dir:
        base_dir = os.path.join(config.base_dir, "data_collection")
    return os.path.join(
        base_dir,
        f"{agent.__name__}_{net.__name__}_{env_name}_{'_'.join(map(str, args))}_{int(time.time())}",
    )


def add_summary_v2(
    summary_writer: tf.summary.SummaryWriter,
    summaries: list,
    step: int,
    flush: bool = False,
) -> bool:
    with summary_writer.as_default():
        for summ_type, *summ_args in summaries:
            status = getattr(tf.summary, summ_type)(*summ_args, step=step)
    if flush:
        summary_writer.flush()
    return status


def add_aim_values(run_l: Run, reports, step):
    for tag, val in reports:
        run_l.track(
            val,
            name=tag,
            step=step,
            context={
                "subset": "train",
            },
        )


def sample_replay_buffer(
    memory: Union[
        circular_replay_buffer.OutOfGraphReplayBuffer,
        offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
    ],
    batch_size: int = None,
    indices: int = None,
) -> Dict[str, jnp.DeviceArray]:
    return dict(
        zip(
            [el.name for el in memory.get_transition_elements(batch_size=batch_size)],
            memory.sample_transition_batch(batch_size=batch_size, indices=indices),
        )
    )
