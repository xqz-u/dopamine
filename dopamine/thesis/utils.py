import functools as ft
import inspect
import logging
import os
import time
from collections import OrderedDict
from itertools import groupby
from typing import Any, Tuple, Union

import attr

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


# default folder structure:
# basedir/ENVIRONMENT/AGENT/exp_name
# the paths in caps lock can be omitted with build_hierarchy=True
def data_dir_from_conf(
    exp_name: str, conf: dict, basedir: str = None, build_hierarchy: bool = True
) -> str:
    intermediate_dirs = (
        ""
        if not build_hierarchy
        else os.path.join(
            "-".join(conf["env"].values()), conf["agent"]["call_"].__name__
        )
    )
    full_path = os.path.join(
        basedir or config.data_dir,
        intermediate_dirs,
        exp_name,
    )
    os.makedirs(full_path, exist_ok=True)
    conf["runner"]["base_dir"] = full_path
    return full_path


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


# precedence: update_dict, all_values
def inplace_dict_assoc(
    d: OrderedDict,
    fn: callable,
    *all_values,
    update_dict: dict = None,
):
    assert any(
        [update_dict, all_values]
    ), "one of update_dict or all_values must not be None"
    if update_dict:
        d.update({k: fn(d[k], v) for k, v in update_dict.items()})
    else:
        for (k, v), nv in zip(d.items(), all_values):
            d[k] = fn(v, nv)
