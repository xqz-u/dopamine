import functools as ft
import time
from itertools import groupby
from typing import Sequence

import jax


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
