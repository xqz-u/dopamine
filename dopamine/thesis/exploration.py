import functools as ft
from typing import Tuple, Union

import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from thesis import custom_pytrees


# taken from dopamine.jax.agents.dqn.dqn_agent
# NOTE intermediate version with default arguments, since defaults args
# in jitted functions behave like static_argnums and are recompiled each
# time they change - problematic with egreedy_linear_decay
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
# when record_max_q is True, the model perfroms an evaluation of the
# state irrespective of exploration or exploitation; on the other
# hand, when exploring and !record_max_q, forward pass on the model is
# performed
@ft.partial(jax.jit, static_argnums=(0, 1, 2))
def egreedy_base(
    net: nn.Module,
    num_actions: int,
    record_max_q: bool,
    epsilon: Union[jnp.ndarray, float],
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.ndarray,
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.ndarray, jnp.ndarray]:
    def greedy():
        qvals = net.apply(params, state)
        return jnp.array([qvals.argmax(), qvals.max()])

    def explore():
        maxq = jnp.where(record_max_q, net.apply(params, state).max(), jnp.nan)
        return jnp.array([jrand.randint(next(rng), (), 0, num_actions), maxq])

    action, maxq = jnp.where(jrand.uniform(next(rng)) <= epsilon, explore(), greedy())
    return rng, action.astype("int32"), maxq


# NOTE accepts **args so that this function and egreedy_linear_decay can
# be called the same way without binding unknown arguments at runtime...
def egreedy(
    net: nn.Module,
    num_actions: int,
    record_max_q: bool,
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.DeviceArray,
    eval_mode: bool = False,
    epsilon_train: float = 0.01,
    epsilon_eval: float = 0.001,
    **_
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.ndarray, jnp.ndarray]:
    return egreedy_base(
        net,
        num_actions,
        record_max_q,
        epsilon_train if not eval_mode else epsilon_eval,
        rng,
        params,
        state,
    )


def egreedy_linear_decay(
    net: nn.Module,
    num_actions: int,
    record_max_q: bool,
    rng: custom_pytrees.PRNGKeyWrap,
    params: FrozenDict,
    state: jnp.DeviceArray,
    eval_mode: bool = False,
    epsilon_train: float = 0.01,
    epsilon_eval: float = 0.001,
    training_steps: int = 0,
    decay_period: int = 250000,
    warmup_steps: int = 500,
    **_
) -> Tuple[custom_pytrees.PRNGKeyWrap, jnp.ndarray, jnp.ndarray]:
    return egreedy_base(
        net,
        num_actions,
        record_max_q,
        epsilon_eval
        if eval_mode
        else linearly_decaying_epsilon(
            decay_period, training_steps, warmup_steps, epsilon_train
        ),
        rng,
        params,
        state,
    )
