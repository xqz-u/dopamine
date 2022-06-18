import functools as ft
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple

import gym
import jax
import numpy as np


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


def callable_name_getter(call_: callable) -> str:
    return getattr(call_, "__name__", type(call_).__name__)


# make a simple configuration reportable easily e.g. in Aim
def reportable_config(conf: dict) -> dict:
    return jax.tree_map(lambda n: callable_name_getter(n) if callable(n) else n, conf)


# recursively create a dict of the form {key: reportable_object.key},
# where a key is given by `reportable_object.fields_specifier`. each
# value in `fields_specifier` can either be:
# - a string to a field, to perform getattr(reportable_object, key)
# - a string to a field with the `fields_specifier` attribute itself:
#   recur and gather the attribute's values
# - a tuple T of the form (key, callable with no parameters), such that
#   {T[0]: T[1]()}
def config_collector(reportable_object: object, fields_specifier: str) -> dict:
    def inner(obj, conf_dict):
        reportables = getattr(obj, fields_specifier, None)
        if not reportables:
            return conf_dict
        for field in reportables:
            if isinstance(field, tuple):
                assert len(field) == 2 and callable(field[1])
                conf_dict[field[0]] = field[1]()
                continue
            value = getattr(obj, field)
            if hasattr(value, fields_specifier):
                conf_dict[field] = {
                    "call_": callable_name_getter(value),
                    **inner(value, {}),
                }
            else:
                conf_dict[field] = value
        return conf_dict

    return inner(reportable_object, {})


# default folder structure:
# basedir/ENVIRONMENT/AGENT/exp_name
# the paths in caps lock can be omitted with build_hierarchy=True
def data_dir_from_conf(
    exp_name: str,
    env_name: str,
    agent_type_name: str,
    basedir: str,
    build_hierarchy: bool = True,
) -> str:
    full_path = os.path.join(
        basedir,
        "" if not build_hierarchy else os.path.join(env_name, agent_type_name),
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
    return np.sum(
        np.repeat([rewards.get(env.spec.name, np.nan)], max_steps) * exponential_gammas
    )
