import functools as ft
import inspect
import logging
import os
import time
from itertools import groupby
from typing import Any, Tuple, Union

import attr
import gin

from thesis import config


def is_builtin(elt) -> bool:
    return elt.__class__.__module__ == "builtins"


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


def attr_fields_d(attr_class: object) -> dict:
    return {
        field.name: getattr(attr_class, field.name)
        for field in attr.fields(type(attr_class))
    }


def argfinder(fn: callable, arg_coll: dict) -> dict:
    return {k: v for k, v in arg_coll.items() if k in inspect.signature(fn).parameters}


# https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
def callable_defaults(elt: callable) -> dict:
    return {
        k: default
        for k, v in inspect.signature(elt).parameters.items()
        if (default := v.default) is not inspect.Parameter.empty
    }


class ConsoleLogger(logging.Logger):
    level: int
    name: str

    def __init__(self, level: int = logging.DEBUG, name: str = None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s:%(levelname)s:%(name)s\n%(message)s",
            datefmt="%m/%d/%Y %I:%M:%S",
        )
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.setLevel(level)


def bind_instance_method(instance: object, name: str, method: callable):
    setattr(instance, name, lambda *args, **kwargs: method(instance, *args, **kwargs))


# def add_summary_v2(
#     summary_writer: tf.summary.SummaryWriter,
#     summaries: list,
#     step: int,
#     flush: bool = False,
# ) -> bool:
#     with summary_writer.as_default():
#         for summ_type, *summ_args in summaries:
#             status = getattr(tf.summary, summ_type)(*summ_args, step=step)
#     if flush:
#         summary_writer.flush()
#     return status
