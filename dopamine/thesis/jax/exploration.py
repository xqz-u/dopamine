import functools as ft
from typing import Tuple, Union

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from thesis import custom_pytrees

import jax
from jax import numpy as jnp
from jax import random as jrand

# TODO it would be better to specify only the args required by
# `linearly_decaying_epsilon` here, and pass the base egreedy args like
# **base_egreedy. This is not supported by thesis.utils.argfinder though
# What I can do is in intermediate abstraction, like a dataclass or
# another map, to encapsulate the common arguments
# NOTE another commonality is that all egreedy functions require net
# and params separated, take this into account in agent code


# taken from dopamine.jax.agents.dqn.dqn_agent
@ft.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(
    decay_period: int, training_steps: int, warmup_steps: int, epsilon_target: float
) -> jnp.DeviceArray:
    steps_left = decay_period + warmup_steps - training_steps
    bonus = (1.0 - epsilon_target) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0.0, 1.0 - epsilon_target)
    return epsilon_target + bonus


# NOTE not passing the whole custom_pytrees.NetworkOptimWrap here
# since this function only needs net and params
@ft.partial(jax.jit, static_argnums=(0, 1))
def egreedy_base(
    net: nn.Module,
    num_actions: int,
    epsilon: Union[jnp.DeviceArray, float],
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.DeviceArray,
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.DeviceArray]:
    return rng, jnp.where(
        jrand.uniform(next(rng)) <= epsilon,
        jrand.randint(next(rng), (), 0, num_actions),
        jnp.argmax(net.apply(params, state)),
    )


# NOTE intermediate version with default arguments, since defaults args
# in jitted functions behave like static_argnums and get recompiled each
# time they change (so could be a problem with egreedy_linear_decay)
def egreedy(
    net: nn.Module,
    num_actions: int,
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.DeviceArray,
    eval_mode: bool = False,
    epsilon_train: float = 0.01,
    epsilon_eval: float = 0.001,
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.DeviceArray]:
    epsilon = epsilon_train if not eval_mode else epsilon_eval
    return egreedy_base(net, num_actions, epsilon, rng, params, state)


def egreedy_linear_decay(
    net: nn.Module,
    num_actions: int,
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.DeviceArray,
    training_steps: int,
    decay_period: int = 250000,
    warmup_steps: int = 500,
    eval_mode: bool = False,
    epsilon_train: float = 0.01,
    epsilon_eval: float = 0.01,
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.DeviceArray]:
    epsilon = (
        epsilon_eval
        if eval_mode
        else linearly_decaying_epsilon(
            decay_period, training_steps, warmup_steps, epsilon_train
        )
    )
    return egreedy_base(net, num_actions, epsilon, rng, params, state)
