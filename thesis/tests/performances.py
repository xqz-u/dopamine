#!/usr/bin/env python3

from typing import Tuple

import jax
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from thesis import utils as u
from thesis.jax import networks


def egreedy_action_selection(
    rng: jnp.DeviceArray,
    epsilon: float,
    num_actions: int,
    q_net: networks.ClassicControlDNNetwork,
    params: FrozenDict,
    state: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    key, key1, key2 = u.force_devicearray_split(rng, 3)
    return key, jnp.where(
        jrand.uniform(key1) <= epsilon,
        jrand.randint(key2, (), 0, num_actions),
        jnp.argmax(q_net.apply(params, state)),
    )


egreedy_jitted = jax.jit(egreedy_action_selection, static_argnums=(1, 2, 3))


def test(selection_fn):
    rng = jrand.PRNGKey(42)
    rng, k = jrand.split(rng)
    state = jrand.uniform(k, (4, 1))
    net = networks.ClassicControlDNNetwork(output_dim=2)
    rng, k = jrand.split(rng)
    params = net.init(k, state)
    rng, action = selection_fn(rng, 0.01, 2, net, params, state)
    return action
