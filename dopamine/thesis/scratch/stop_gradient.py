import functools as ft
from typing import Sequence

from dopamine.jax import losses
from flax import linen as nn

import jax
from jax import numpy as jnp
from jax import random as jrand


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __hash__(self):
        return ft.reduce(lambda acc, x: hash(hash(x) + acc), self.layers, 0)


rng = jrand.PRNGKey(0)

rng, key = jrand.split(rng)
state = jrand.uniform(key, (4,))

model = Sequential([nn.Dense(features=2)])
rng, key = jrand.split(rng)
params = model.init(key, state)

# jax.tree_map(jnp.shape, params)


def td_target(net, params, state, reward, terminal, gamma=1.0):
    return reward + gamma * net.apply(params, state).max() * (1.0 - terminal)


def td_target_nograd(net, params, state, reward, terminal, gamma=1.0):
    return jax.lax.stop_gradient(
        reward + gamma * net.apply(params, state).max() * (1.0 - terminal)
    )


def loss_fn(params, x, y_hat, action):
    y = model.apply(params, x)[action]
    return losses.mse_loss(y_hat, y)


deriv = jax.value_and_grad(loss_fn)

for i in range(100):
    print(i)
    rng, key, key1, key2 = jrand.split(rng, 4)
    rand_state = jrand.uniform(key, (4,))
    rand_state1p = jrand.uniform(key1, (4,))
    reward = jrand.uniform(key2)
    action = jnp.array(0)
    terminal = jnp.array(0)

    target = td_target(model, params, rand_state1p, reward, terminal)
    loss, grad = deriv(params, rand_state, target, action)

    target_ng = td_target_nograd(model, params, rand_state1p, reward, terminal)
    loss_ng, grad_ng = deriv(params, rand_state, target_ng, action)
    if not all(
        jax.tree_leaves(
            jax.tree_multimap(lambda x, y: jnp.array_equal(x, y), grad, grad_ng)
        )
    ):
        print("diff!")

# there seems to be no difference with calling `jax.lax.stop_gradient`?
