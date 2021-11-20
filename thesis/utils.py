import functools as ft
import time
from itertools import groupby


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def timer(fn):
    @ft.wraps(fn)
    def inner(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        print(f"`{fn.__name__}` exec time: {time.time () - start}")
        return ret

    return inner
