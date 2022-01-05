import functools as ft
from typing import Sequence

import flax
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


net_eval = lambda net, params, inp: net.apply(params, inp)
batch_net_eval = lambda net, params, states: jax.vmap(
    net_eval, in_axes=(None, None, 0)
)(net, params, states)

q_chosen_actions = lambda qs, actions: jax.vmap(lambda v, a: v[a])(qs, actions)


# TODO stop_gradient?
def td_target(net, params, next_states, rewards, terminals, gamma=1.0):
    max_q = batch_net_eval(net, params, next_states).max(1)
    # return rewards + gamma * max_q * (1.0 - terminals)
    return jax.lax.stop_gradient(rewards + gamma * max_q * (1.0 - terminals))


def loss_fn(params, net, xs, y_hats, actions):
    ys = q_chosen_actions(batch_net_eval(net, params, xs), actions)
    return jnp.mean(jax.vmap(losses.mse_loss)(y_hats, ys))


gamma = 0.9
rng, key0, key1, key2, key3, key4 = jrand.split(jrand.PRNGKey(0), 6)
states = jrand.uniform(key0, (10, 4), minval=-1, maxval=1)
next_states = jrand.uniform(key1, (10, 4), minval=-1, maxval=1)
actions = jrand.randint(key2, (10,), 0, 2)
terminals = jrand.randint(key3, (10,), 0, 2)
rewards = jrand.uniform(key4, (10,))


# simple perceptron: w*x + b
model = Sequential([nn.Dense(features=2)])
rng, key = jrand.split(rng)
params = model.init(key, jnp.zeros((4,)))

# jax.tree_map(jnp.shape, params)

targets = td_target(model, params, next_states, rewards, terminals, gamma)
loss_fn(params, model, states, targets, actions)
jax.jit(loss_fn, static_argnums=(1,))(params, model, states, targets, actions)
# jax.make_jaxpr(loss_fn, static_argnums=(1,))(params, model, states, targets, actions)
# jax.make_jaxpr(jax.jit(loss_fn, static_argnums=(1,)), static_argnums=(1))(
#     params, model, states, targets, actions
# )

deriv = jax.value_and_grad(loss_fn)
# jax.make_jaxpr(deriv, static_argnums=(1,))(params, model, states, targets, actions)
loss, grad = deriv(params, model, states, targets, actions)


import optax

opt = optax.adam(learning_rate=0.001)
opt_state = opt.init(params)


# rng, key = jrand.split(rng)
# s_new = states.reshape((10, 4, 1))
# params_r = model.init(key, s_new)

# qv = batch_net_eval(model, params_r, s_new)

# batch_net_eval(model, params_r, s_new).max(1)


# jnp.dot(q, next_states)


def summer(x):
    def inner(y):
        return y * 4

    return x + inner(x - 1)


summer_jit = jax.jit(summer)
jax.make_jaxpr(summer)(states)


def optimize(
    states,
    actions,
    next_states,
    rewards,
    terminals,
    params: flax.core.frozen_dict.FrozenDict,
    target_params: flax.core.frozen_dict.FrozenDict,
    optim_state: optax.OptState,
    net: nn.Module,
    optim: optax.GradientTransformation,
    gamma: float,
):
    def loss_fn(params, targets) -> jnp.DeviceArray:
        qv = q_chosen_actions(batch_net_eval(net, params, states), actions).squeeze()
        return jnp.mean(jax.vmap(losses.huber_loss)(targets, qv))

    target_qs = td_target(net, target_params, next_states, rewards, terminals, gamma)
    deriv = jax.value_and_grad(loss_fn)
    loss, grads = deriv(params, target_qs)
    updates, optim_state = optim.update(grads, optim_state, params=params)
    params = optax.apply_updates(params, updates)
    return optim_state, params, loss

    # def loss_fn(params, target):
    #     def q_online(state):
    #         return net.apply(params, state)

    #     q_values = jax.vmap(q_online)(states)
    #     q_values = jnp.squeeze(q_values)
    #     replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    #     return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))

    # def q_target(state):
    #     return q_net.apply(target_params, state.squeeze())

    # target = target_q(q_target, next_states, rewards, terminals, gamma)
    # grad_fn = jax.value_and_grad(loss_fn)
    # loss, grad = grad_fn(params, target)
    # updates, optimizer_state = optim.update(grad, optim_state, params=params)
    # online_params = optax.apply_updates(params, updates)
    # return optimizer_state, online_params, loss


optimize(
    states,
    actions,
    next_states,
    rewards,
    terminals,
    params,
    params,
    opt_state,
    model,
    opt,
    gamma,
)

# @ft.partial(jax.jit, static_argnums=(8, 9, 10))
jax.make_jaxpr(optimize, static_argnums=(8, 9, 10))(
    states,
    actions,
    next_states,
    rewards,
    terminals,
    params,
    params,
    opt_state,
    model,
    opt,
    gamma,
)


@ft.partial(jax.jit, static_argnums=(8, 9, 10))
def optimize(
    states,
    actions,
    next_states,
    rewards,
    terminals,
    params: flax.core.frozen_dict.FrozenDict,
    target_params: flax.core.frozen_dict.FrozenDict,
    opt_state: optax.OptState,
    net: nn.Module,
    opt: optax.GradientTransformation,
    gamma: float,
):
    def loss_fn(params, targets) -> jnp.DeviceArray:
        q_vs = jax.vmap(lambda s: net.apply(params, s))(states)
        q_actions = jax.vmap(lambda v, a: v[a])(q_vs, actions)
        return jnp.mean(jax.vmap(losses.huber_loss)(targets, q_actions))

    target_qs = td_target(net, target_params, next_states, rewards, terminals, gamma)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, target_qs)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


# def pippo(net, params, states, targets):
#     def loss_fn(params, targets) -> jnp.DeviceArray:
#         qv = q_chosen_actions(batch_net_eval(net, params, states), actions).squeeze()
#         return jnp.mean(jax.vmap(losses.huber_loss)(targets, qv))
#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(params, targets)
#     return loss

# jax.make_jaxpr(jax.jit(pippo, static_argnums=(0)), static_argnums=(0))(model, params, states, targets)


# def pippo_nocalls(net, params, states, targets):
#     def loss_fn(params, targets) -> jnp.DeviceArray:
#         q_vs = jax.vmap(lambda s: net.apply(params, s))(states)
#         q_actions = jax.vmap(lambda v, a: v[a])(q_vs, actions)
#         return jnp.mean(jax.vmap(losses.huber_loss)(targets, q_actions))
#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(params, targets)
#     return loss

# jax.make_jaxpr(jax.jit(pippo_nocalls, static_argnums=(0)), static_argnums=(0))(model, params, states, targets)


@ft.partial(jax.jit, static_argnums=(0, 1, 2))
def optimize_dict(
    net: nn.Module,
    opt: optax.GradientTransformation,
    gamma: float,
    replay_elts: dict,
    # states,
    # actions,
    # next_states,
    # rewards,
    # terminals,
    params: flax.core.frozen_dict.FrozenDict,
    target_params: flax.core.frozen_dict.FrozenDict,
    opt_state: optax.OptState,
):
    def loss_fn(params, targets) -> jnp.DeviceArray:
        q_vs = jax.vmap(lambda s: net.apply(params, s))(replay_elts["state"])
        q_actions = jax.vmap(lambda v, a: v[a])(q_vs, replay_elts["action"])
        return jnp.mean(jax.vmap(losses.huber_loss)(targets, q_actions))

    target_qs = td_target(
        net,
        target_params,
        replay_elts["next_state"],
        replay_elts["reward"],
        replay_elts["terminal"],
        gamma,
    )
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, target_qs)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


d = {
    "state": states,
    "next_state": next_states,
    "terminal": terminals,
    "action": actions,
    "reward": rewards,
}

jax.make_jaxpr(optimize_dict, static_argnums=(0, 1, 2))(
    model, opt, gamma, d, params, params, opt_state
)


@jax.jit
def cip(p, d):
    def inner(p):
        print(d)
        return jnp.sum(jnp.power(p - d, 2))

    g = jax.value_and_grad(inner)
    loss, grads = g(p)
    return loss


@jax.jit
def ciop(p, d):
    def inner(p, d):
        return jnp.sum(jnp.power(p - d, 2))

    g = jax.value_and_grad(inner)
    loss, grads = g(p, d)
    return loss


rng, key0, key1 = jrand.split(rng, 3)
a = jrand.uniform(key0, (4,))
b = jrand.uniform(key1, (4,))

jax.make_jaxpr(cip)(a, b)
jax.make_jaxpr(ciop)(a, b)

c = jrand.uniform(key0, (5,))
d = jrand.uniform(key1, (5,))

cip(a, b)
cip(c, d)
