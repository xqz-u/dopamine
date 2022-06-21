import time

import jax
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, networks, utils


def eval_net(net, params, state):
    return net.apply(params, state)


eval_net_jit = jax.jit(eval_net, static_argnums=(0,))


def time_loop(iterations, fn):
    rng = custom_pytrees.PRNGKeyWrap(5)
    state_shape = (84, 84, 4)

    net = networks.NatureDQNNetwork(num_actions=2)
    params = net.init(next(rng), jnp.zeros(state_shape))
    print(utils.jax_container_shapes(params))

    timer = 0.0
    print(f"Sarting {fn.__name__}")

    for _ in range(iterations):
        start = time.time()
        res = fn(
            net, params, jrand.randint(next(rng), state_shape, minval=0, maxval=256)
        ).block_until_ready()
        timer += time.time() - start
    print(f"mean {iterations} iterations: {timer / iterations}")


print(f"jax device: {jax.default_backend()}")


iters = 100

time_loop(iters, eval_net)
time_loop(iters, eval_net_jit)
