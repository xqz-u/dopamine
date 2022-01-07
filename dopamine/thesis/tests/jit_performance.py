import functools as ft

import jax
import optax
from jax import numpy as jnp
from jax import random as jrand
from memory_profiler import profile
from thesis import jax_utils as jax_u
from thesis import utils as u
from thesis.jax import networks
from thesis.offline.replay_memory import offline_circular_replay_buffer

# ANALYZE:
# - performance: pmap, parallelezitaion; memory footprint, since i'm
#                passing numpy arrays?
#   -> passing numpy array: DON'T do that, up to 1000Xslower! still did
# not profile function call correctly though...
# - complexity: resulting jaxpr, resulting python
#   -> can pass functions to jitted functions as static argnums
#   -> can pass args, kwargs


# @ft.partial(jax.jit, static_argnums=(0, 3, 5, 7))
def train_module(
    net,
    params,
    td_errors: jnp.DeviceArray,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: callable,
    states,
    mask: callable,
    *args,
    **kwargs,
):
    # evaluation and loss function
    def estimate_states(params):
        # print(params)
        # print(states)
        # print(td_errors)
        estimates = jax.vmap(lambda state: net.apply(params, state))(states)
        estimates = mask(estimates.squeeze(), *args, **kwargs)
        return jnp.mean(jax.vmap(loss_fn)(td_errors, estimates))

    # optimize the network, taking the gradient of the loss function
    grad_fn = jax.value_and_grad(estimate_states)
    loss, grad = grad_fn(params)
    updates, opt_state = optim.update(grad, opt_state, loss)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


@ft.partial(jax.jit, static_argnums=(0, 5, 6))
def dqv_family_td_error(
    net,
    target_params,
    next_states,
    rewards,
    terminals,
    gamma: float,
    mask: callable,
    *args,
    **kwargs,
) -> jnp.DeviceArray:
    values = jax.vmap(lambda state: net.apply(target_params, state))(next_states)
    values = mask(values, *args, **kwargs)
    terminals_mask = jnp.expand_dims(terminals.squeeze(), 1)
    return jax.lax.stop_gradient(rewards + gamma * values * (1 - terminals))


def mse(x, y):
    return jnp.power(x - y, 2)


state_shape = (4,)
n_obs = 10
batch_state_shape = (n_obs,) + state_shape

key = jax_u.PRNGKeyWrap(0)
states = jrand.uniform(next(key), batch_state_shape, minval=-1.0)
next_states = jrand.uniform(next(key), batch_state_shape, minval=-1.0)
actions = jrand.randint(next(key), (n_obs,), 0, 2)
rewards = jrand.uniform(next(key), (n_obs,))
terminals = jrand.randint(next(key), (n_obs,), 0, 2)

net = networks.ClassicControlDNNetwork(output_dim=2)
params = net.init(next(key), states[0])

opt = optax.adam(learning_rate=0.001)
opt_state = opt.init(params)


def max_action_mask(arr):
    return arr.max(1)


td_err_args = [net, params, next_states, rewards, terminals, 0.99, max_action_mask]
jax.make_jaxpr(dqv_family_td_error, static_argnums=(0, 5, 6))(*td_err_args)

td_errors = dqv_family_td_error(*td_err_args)

# n_params, n_opt_state, loss = train_module(
#     net,
#     params,
#     td_errors,
#     opt,
#     opt_state,
#     mse,
#     states,
#     dqv_base.mask_q_estimates,
#     actions,
# )


path = "/home/xqz-u/uni/dopamine/resources/data/replay_buffer_sample"
buff = offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer(
    (4, 1), stack_size=1, batch_size=32, observation_dtype=states[0].dtype
)
buff.load_buffers(path, [0])
elts = u.sample_replay_buffer(buff)


def evaller(net, params, d, fn=dqv_family_td_error):
    return fn(
        net,
        params,
        d["next_state"],
        d["reward"],
        d["terminal"],
        0.99,
        max_action_mask,
    ).block_until_ready()


t_evaller = u.timer(evaller)
prof_evaller = profile(evaller)


@u.timer
def convert(x):
    return jax.tree_map(jnp.array, x)


print("bellman target comp. numpy")
t_evaller(net, params, elts)
# prof_evaller(net, params, elts)

# negligible overhead
elts_jnp = convert(elts)
print("bellman target comp. jnp")
t_evaller(net, params, elts_jnp)
# prof_evaller(net, params, elts_jnp)


@ft.partial(jax.jit, static_argnums=(0, 5, 6))
def dqv_family_td_error_pmap(
    net,
    target_params,
    next_states,
    rewards,
    terminals,
    gamma: float,
    mask: callable,
    *args,
    **kwargs,
) -> jnp.DeviceArray:
    values = jax.soft_pmap(lambda state: net.apply(target_params, state))(next_states)
    values = mask(values, *args, **kwargs)
    terminals_mask = jnp.expand_dims(terminals.squeeze(), 1)
    return jax.lax.stop_gradient(rewards + gamma * values * (1 - terminals))


def mean_exec_time(n, fn, buff, net, params, bs=500, convert=False):
    evaller_mean_t = u.timer(True)(evaller)
    avg_s = 0
    for i in range(n):
        more_elts = u.sample_replay_buffer(buff, batch_size=bs)
        if convert:
            more_elts = jax.tree_map(jnp.array, more_elts)
        _, t = evaller_mean_t(net, params, more_elts, fn=fn)
        avg_s = (avg_s + t) / 2
    print(f"mean exec time in {n} reps: {avg_s}s")


# only slightly faster?! there was a
# greater speedup with just one call!
print("np: ")
print(mean_exec_time(100, dqv_family_td_error, buff, net, params))
print("jnp: ")
print(mean_exec_time(100, dqv_family_td_error, buff, net, params, convert=True))
# FIXME does not work? need more work, see https://github.com/google/jax/discussions/4198
# print(mean_exec_time(100, dqv_family_td_error_pmap, buff, net, params))
