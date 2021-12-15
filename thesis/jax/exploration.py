#!/usr/bin/env python3

import functools as ft
from typing import Tuple

import flax
import jax
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrand

from thesis import utils as u


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
    return key, jnp.where(
        jrand.uniform(key1) <= epsilon,
        jrand.randint(key2, (), 0, num_actions),
        jnp.argmax(q_net.apply(params, state)),
    )
