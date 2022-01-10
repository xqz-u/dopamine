# NOTE keeping rngs on their own and splitting them as necessary is
# faster than using my custom PyTree generator class. For now I will
# keep using because it provides a cleaner interface, but it ideally
# needs improvements. Look into https://cgarciae.github.io/treeo/ or
# flax.struct for some alternatives; also, I do not know if my class
# is the proper Jax way to accomplish the design I want.

import functools as ft

import jax
from jax import numpy as jnp
from jax import random as jrand


def show_example(structured):
    flat, tree = jax.tree_util.tree_flatten(structured)
    unflattened = jax.tree_util.tree_unflatten(tree, flat)
    print(
        "structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
            structured, flat, tree, unflattened
        )
    )


# wrong! a is unchanged at top level
@jax.jit
def cip(a):
    e = jax.random.uniform(next(a), (4,))
    c = jax.random.uniform(next(a), (4,))
    return e, c


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def ciop(rng, eps, num_actions, net, params, state):
    rng, k1, k2 = jrand.split(rng, 3)
    return rng, jnp.where(
        jrand.uniform(k1) <= eps,
        jrand.randint(k2, (), 0, num_actions),
        jnp.argmax(net.apply(params, state)),
    )


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def pippo(rng, eps, num_actions, net, params, state):
    # print(f"changed: {a}")
    return rng, jnp.where(
        jrand.uniform(next(rng)) <= eps,
        jrand.randint(next(rng), (), 0, num_actions),
        jnp.argmax(net.apply(params, state)),
    )


def time_pippo(k, net, params):
    for i in range(100):
        state = jrand.uniform(next(k), (4,))
        k, _ = pippo(k, 0.01, 2, net, params, state)


def time_ciop(k, net, params):
    for i in range(100):
        k, sk = jrand.split(k)
        state = jrand.uniform(sk, (4,))
        k, _ = ciop(k, 0.01, 2, net, params, state)


# import timeit
# from thesis import PRNGKeyWrap
# from thesis.jax import networks
# net = networks.mlp(2, [512, 512])
# k = PRNGKeyWrap.PRNGKeyWrap(0)
# nk = jrand.PRNGKey(0)
# params = net.init(next(k), jnp.ones((4,)))

# pt = timeit.timeit("time_pippo(k, net, params)", number=10000, globals=locals())
# ct = timeit.timeit("time_ciop(nk, net, params)", number=10000, globals=locals())

# k, _ = pippo(k, 0.01, 2, net, params, jnp.ones((4,)))

# k = PRNGKeyWrap.PRNGKeyWrap(5)
# show_example(k)
# a, b, k = pippo(k)
# print(k)
# c, d, k = pippo(k)
# print(k)
# print(jax.make_jaxpr(pippo)(k))
# print(jax.make_jaxpr(cip)(k))
# print(jax.make_jaxpr(ciop)(jax.random.PRNGKey(0)))
# a, b = cip(k)
# k
