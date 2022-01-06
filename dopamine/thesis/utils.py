import functools as ft
import os
import time
from itertools import groupby
from typing import Sequence

import gin
import tensorflow as tf
from aim import Run

import jax
from thesis import config


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# TODO use timeit module to have proper execution time metrics
def timer(fn):
    @ft.wraps(fn)
    def inner(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        print(f"`{fn.__name__}` exec time: {time.time () - start}")
        return ret

    return inner


def dict_pop(d, k):
    return (d, d.pop(k))[0]


def mget(d, *keys):
    return [d[k] for k in keys]


def force_devicearray_split(
    key: jax.random.PRNGKey, n=2
) -> Sequence[jax.random.PRNGKey]:
    splits = jax.random.split(key, n)
    return [jax.numpy.asarray(k) for k in splits]


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
            epoch=0,
            context={
                "subset": "train",
            },
        )
