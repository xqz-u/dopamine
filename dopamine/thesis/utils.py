import functools as ft
import logging
import math
import os
from itertools import groupby
from pathlib import Path
from typing import List, Tuple

import gym
import jax
import numpy as np

from thesis import constants


# set root logging level to be the lowest one; submodule can decide
# which messages to ignore
def setup_root_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="(PID=%(process)s) [%(asctime)s] [%(levelname)-8s] -- %(message)s -- (%(name)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def jax_container_shapes(cont) -> dict[str, Tuple[int]]:
    return jax.tree_map(lambda el: el.shape, cont)


def is_builtin(elt) -> bool:
    return elt.__class__.__module__ == "builtins"


def reportable_conf(conf: dict) -> dict:
    return jax.tree_map(
        lambda v: f"<{v.__name__}>"
        if callable(v)
        else (str(v) if not is_builtin(v) else v),
        conf,
    )


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


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
            f"{conf['env']['environment_name']}-{conf['env']['version']}",
            conf["agent"]["call_"].__name__,
        )
    )
    full_path = os.path.join(
        basedir or constants.data_dir,
        intermediate_dirs,
        exp_name,
    )
    os.makedirs(full_path, exist_ok=True)
    return full_path


def list_all_ckpt_iterations(base_directory: str) -> List[int]:
    iters = [
        int(f.name.split(".gz")[0].split(".")[1])
        for f in Path(base_directory).glob("add_count_ckpt.*.gz")
    ]
    return list(range(min(iters), max(iters) + 1))


# NOTE sorted uses alphanumeric ordering by default
# this function finds replay buffers stored in (possibly) multiple
# folders starting from 1 level of depth from base_dir; a directory
# structure of arbitrary depth can exist after this level, so 2
# 2 folder structure are admissible:
# /base_dir
#   /replay_buffers_dir_0
#     /inter_tree (of arbitrary depth, optional)
#       /replay_buffers_files_0
#   /replay_buffers_dir_1
#     /inter_tree (of arbitrary depth, optional)
#       /replay_buffers_files_1
# ...
def unfold_replay_buffers_dir(base_dir: str, inter_tree: str = "") -> List[str]:
    return [os.path.join(base_dir, d, inter_tree) for d in sorted(os.listdir(base_dir))]


# TODO add Pendulum, which has a formula to compute reward
@ft.lru_cache
def deterministic_discounted_return(env: gym.Env, discount: float = 0.99) -> float:
    """
    Computes the discounted return G_t for a fully deterministic problem
    - gym's classic control environments - according to:

    \[ G_t = R_{t+1}+ \gamma R_{t+2}+ \gamma R^2_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^kR_{t+k+1} \]

    Supported environments: CartPole, Acrobot, MountainCar (discrete).
    """
    rewards = {"CartPole": 1, "Acrobot": -1, "MountainCar": -1, "Pendulum": ...}
    max_steps = env.spec.max_episode_steps
    exponential_gammas = np.array([math.pow(discount, k) for k in range(max_steps)])
    print("called!")
    return np.sum(
        np.repeat([rewards.get(env.spec.name, np.nan)], max_steps) * exponential_gammas
    )


# def attr_method_binder(self, attribute, value):
#     setattr(self, attribute.name, types.MethodType(value, self))
# def timer(want_time=False):
#     def decorator(fn):
#         @ft.wraps(fn)
#         def inner(*args, **kwargs) -> Union[Any, Tuple[Any, float]]:
#             start = time.time()
#             ret = fn(*args, **kwargs)
#             end = time.time() - start
#             if not want_time:
#                 print(f"`{fn.__name__}` exec time: {end}")
#             return ret if not want_time else (ret, end)
#         return inner
#     # trick to keep writing @timer, @timer() and @timer(True)
#     if callable(want_time):
#         f = want_time
#         want_time = False
#         return decorator(f)
#     return decorator
# def attr_fields_d(attr_inst: object, get_props: bool = False) -> dict:
#     attr_class = type(attr_inst)
#     return {
#         **{
#             field.name: getattr(attr_inst, field.name)
#             for field in attr.fields(attr_class)
#         },
#         **(
#             {}
#             if not get_props
#             else {
#                 p: getattr(attr_inst, p)
#                 for p in dir(attr_class)
#                 if isinstance(getattr(attr_class, p), property)
#             }
#         ),
#     }
# def argfinder(fn: callable, arg_coll: dict) -> dict:
#     return {k: v for k, v in arg_coll.items() if k in inspect.signature(fn).parameters}
# # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
# def callable_defaults(elt: callable) -> dict:
#     return {
#         k: default
#         for k, v in inspect.signature(elt).parameters.items()
#         if (default := v.default) is not inspect.Parameter.empty
#     }
# # precedence: update_dict, all_values
# def inplace_dict_assoc(
#     d: OrderedDict,
#     fn: callable,
#     *all_values,
#     update_dict: dict = None,
# ):
#     assert any(
#         [update_dict, all_values]
#     ), "one of update_dict or all_values must not be None"
#     if update_dict:
#         d.update({k: fn(d[k], v) for k, v in update_dict.items()})
#     else:
#         for (k, v), nv in zip(d.items(), all_values):
#             d[k] = fn(v, nv)
