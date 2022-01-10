import functools as ft
from typing import Tuple

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from thesis import custom_pytrees

import jax
from jax import numpy as jnp
from jax import random as jrand


# NOTE taken from dopamine.jax.agents.dqn.dqn_agent
@ft.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(
    decay_period: int, training_steps: int, warmup_steps: int, epsilon_target: float
) -> jnp.DeviceArray:
    steps_left = decay_period + warmup_steps - training_steps
    bonus = (1.0 - epsilon_target) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0.0, 1.0 - epsilon_target)
    return epsilon_target + bonus


# TODO need an intermediate version with kwargs to have defaults
# in the config
@ft.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def egreedy(
    num_actions: int,
    eval_mode: bool,
    epsilon_train: float,
    epsilon_eval: float,
    state: jnp.DeviceArray,
    qnet: custom_pytrees.NetworkOptimWrap,
    rng: custom_pytrees.PRNGKeyWrap,
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.DeviceArray]:
    epsilon = epsilon_train if not eval_mode else epsilon_eval
    return rng, jnp.where(
        jrand.uniform(next(rng)) <= epsilon,
        jrand.randint(next(rng), (), 0, num_actions),
        jnp.argmax(qnet.net.apply(qnet.params["online"], state)),
    )


# TODO rewrite, when using linear decay on epsilon it cannot be marked
# as static_argnum by jitted functions, since it will always be
# different until decay_period has occurred -> a lot of useless caching!
# NOTE it would be better to specify only the args required by
# `linearly_decaying_epsilon` here, and pass the base egreedy args like
# **base_egreedy. This is not supported by thesis.utils.argfinder though
def egreedy_linear_decay(
    training_steps: int = None,
    decay_period: int = 250000,
    warmup_steps: int = 500,
    qnet: custom_pytrees.NetworkOptimWrap = None,
    rng: custom_pytrees.PRNGKeyWrap = None,
    num_actions: int = None,
    eval_mode: bool = False,
    epsilon_train: float = 0.01,
    epsilon_eval: float = 0.01,
    state: jnp.DeviceArray = None,
) -> jnp.DeviceArray:
    if not eval_mode:
        epsilon_train = linearly_decaying_epsilon(
            decay_period, training_steps, warmup_steps, epsilon_train
        )

    return egreedy(
        num_actions, eval_mode, epsilon_train, epsilon_eval, state, qnet, rng
    )


# NOTE not passing the whole custom_pytrees.NetworkOptimWrap here
# since this function only needs net and params
# @ft.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
# def egreedy(
#     net: nn.Module,
#     num_actions: int,
#     eval_mode: bool,
#     epsilon_train: float,
#     epsilon_eval: float,
#     rng: custom_pytrees.PRNGKeyWrap,
#     params: FrozenDict,
#     state: jnp.DeviceArray,
# ) -> jnp.DeviceArray:
#     epsilon = epsilon_train if not eval_mode else epsilon_eval
#     return rng, jnp.where(
#         jrand.uniform(next(rng)) <= epsilon,
#         jrand.randint(next(rng), (), 0, num_actions),
#         jnp.argmax(net.apply(params, state)),
#     )
