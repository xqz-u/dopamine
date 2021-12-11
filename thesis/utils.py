import functools as ft
import os
import time
from itertools import groupby
from typing import Sequence

import gin
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


def mget(d, keys):
    return [d[k] for k in keys]


def force_devicearray_split(
    key: jax.random.PRNGKey, n=2
) -> Sequence[jax.random.PRNGKey]:
    splits = jax.random.split(key, n)
    return [jax.numpy.asarray(k) for k in splits]


# experiment_spec = [agent name, network name, environment name]
@gin.configurable
def make_unique_data_dir(experiment_spec: list, base_dir: str = None) -> str:
    agent, net, env_name = experiment_spec
    if not base_dir:
        base_dir = os.path.join(config.base_dir, "online", "data_collection")
    return os.path.join(
        base_dir, f"{agent.__name__}_{net.__name__}_{env_name}_{int(time.time())}"
    )
