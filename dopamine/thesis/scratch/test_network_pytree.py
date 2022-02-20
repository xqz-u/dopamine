import jax
import optax
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees
from thesis.jax import networks
from thesis.tests import test_rng_pytree


@jax.jit
def pippone(a):
    a.params = jax.tree_map(lambda p: p * 2, a.params)
    return a


@jax.jit
def cip(a):
    return a


@jax.jit
def real_routine(params_opt: custom_pytrees.NetworkOptimWrap, state, target):
    def loss_fn(params):
        estim = params_opt.net.apply(params, state)
        return jnp.mean(jnp.power(target - estim, 2))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params_opt.params)
    updates, params_opt.opt_state = params_opt.optim.update(
        grads, params_opt.opt_state, params=params_opt.params
    )
    params_opt.params = optax.apply_updates(params_opt.params, updates)
    return params_opt, loss


# net = networks.mlp(2, [4])
# k = custom_pytrees.PRNGKeyWrap(0)
# params = net.init(next(k), jnp.ones((4,)))
# opt = optax.adam(learning_rate=0.01)
# opt_state = opt.init(params)


# c = custom_pytrees.NetworkOptimWrap(net, opt, params, opt_state)


# target = jnp.ones((2,))
# state = jax.random.randint(next(k), (4,), 0, 3)

# p, loss0 = real_routine(c, state, target)
# p, loss1 = real_routine(c, state, target)

# v, loss2 = real_routine(p, state, target)
