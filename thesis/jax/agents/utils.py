#!/usr/bin/env python3

import jax
import numpy as onp
from jax import numpy as jnp


def replay_chosen_q(
    estimates: jnp.DeviceArray, replay_actions: onp.ndarray
) -> jnp.DeviceArray:
    """
    Given Q-values (a matrix of shape (replayed_states, n_actions)),
    extract the Q-values of the action chosen for replay and indexed
    by `replay_actions`.
    """
    return jax.vmap(lambda x, y: x[y])(estimates, replay_actions)
