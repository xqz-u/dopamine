#!/usr/bin/env python3

import functools as ft
from typing import Tuple

import flax
from flax import linen as nn
from thesis import utils as u

import jax
from jax import numpy as jnp
from jax import random as jrand


@ft.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def egreedy_action_selection(
    rng: jnp.DeviceArray,
    q_net: nn.Module,
    num_actions: int,
    eval_mode: bool,
    epsilon_train: float,
    epsilon_eval: float,
    params: flax.core.frozen_dict.FrozenDict,
    state: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    epsilon = epsilon_train if not eval_mode else epsilon_eval
    key, key1, key2 = u.force_devicearray_split(rng, 3)
    # p = jrand.uniform(key1)
    # if p <= epsilon:
    #     print("RAND act!")
    #     ret = jrand.randint(key2, (), 0, num_actions)
    # else:
    #     v = q_net.apply(params, state)
    #     ret = jnp.argmax(v)
    #     print(f"GREEDY act! raw: {v}, i: {ret}")
    # return key, ret
    return key, jnp.where(
        jrand.uniform(key1) <= epsilon,
        jrand.randint(key2, (), 0, num_actions),
        jnp.argmax(q_net.apply(params, state)),
    )
