import functools as ft

import jax
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees


# @ft.partial(jax.jit, static_argnums=())
@jax.jit
def egreedy(rng: custom_pytrees.PRNGKeyWrap, eps: float):
    print("change")
    return rng, jnp.where(
        jrand.uniform(next(rng)) <= eps,
        jrand.randint(next(rng), (), 0, 2),
        jrand.uniform(next(rng), (4,)),
    )


# @jax.jit
# def egreedy_better(rng: custom_pytrees.PRNGKeyWrap, eps: float):
#     if jrand.uniform(next(rng)) <= eps:
#         return rng, jrand.randint(next(rng), 0, 2)
#     return rng, jrand.uniform(next(rng), (4,))


def identity_epsilon(eps: float, *_):
    return eps


def epsilon_delta(eps: float, rng):
    return float(eps + jrand.uniform(next(rng)))


k = custom_pytrees.PRNGKeyWrap(4)
eps = 0.01
eps_d = epsilon_delta(eps, k)
# k, res = egreedy_better(k, eps_d)
# k, res = egreedy(k, eps_d)
# eps_d = epsilon_delta(eps, k)
# k, res = egreedy(k, eps_d)
# eps_d = epsilon_delta(eps, k)
# k, res = egreedy(k, eps_d)
# eps_d = epsilon_delta(eps, k)
# k, res = egreedy(k, eps_d)


# jnp.where(
#     jrand.uniform(next(k)) <= eps,
#     jrand.randint(next(k), (), 0, 2),
#     jrand.uniform(next(k), (4,)),
# )


@jax.jit
def pippo(a, b=jnp.ones((4,))):
    print("change")
    return a + b


@jax.jit
def pippa(a, b):
    print("change")
    return a + b


# pippo(jax.numpy.ones((4,)))
# pippo(jax.numpy.ones((4,)))
# pippo(jax.numpy.ones((4,)), jnp.zeros((4,)))
# pippo(jax.numpy.ones((4,)), jnp.zeros((4,)))

# x = jax.numpy.ones((4,))
# y = jax.numpy.zeros_like(x)
# pippa(x, x)
# pippa(x, x)
# pippa(x, y)
# pippa(x, y)
