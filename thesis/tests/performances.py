#!/usr/bin/env python3

from typing import Tuple

import jax
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand

from thesis import experiment_data as exp_data
from thesis import utils as u
from thesis.jax import networks
from thesis.jax.agents import dqv_agent as dqv


def egreedy_action_selection(
    rng: jnp.DeviceArray,
    epsilon: float,
    num_actions: int,
    q_net: networks.ClassicControlDNNetwork,
    params: FrozenDict,
    state: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    print(rng)
    print(epsilon)
    print(num_actions)
    print(q_net)
    print(params)
    print(state)
    key, key1, key2 = u.force_devicearray_split(rng, 3)
    return key, jnp.where(
        jrand.uniform(key1) <= epsilon,
        jrand.randint(key2, (), 0, num_actions),
        jnp.argmax(q_net.apply(params, state)),
    )


egreedy_jitted = jax.jit(egreedy_action_selection, static_argnums=(1, 2, 3))
egreedy_jitted_no_static = jax.jit(egreedy_action_selection, static_argnums=(3,))


# profile action selection
def create_net_and_params():
    rng, k = u.force_devicearray_split(jrand.PRNGKey(42))
    state = jrand.uniform(k, (4, 1))
    net = networks.ClassicControlDNNetwork(output_dim=2)
    rng, k = u.force_devicearray_split(rng)
    params = net.init(k, state)
    return rng, net, params, state


def profile_action_selection(selection_fn, iters, rng, net, params, state):
    chosen_actions = []
    for _ in range(iters):
        rng, action = selection_fn(rng, 0.01, 2, net, params, state)
        chosen_actions.append(action)
        rng, k = u.force_devicearray_split(rng)
        state = jrand.uniform(k, (4, 1))
    return chosen_actions


def test_action_selection():
    # %timeit profile_action_selection(dqv.egreedy_action_selection, 100, *create_net_and_params())
    # %timeit profile_action_selection(dqv.egreedy_action_selection_jit, 100, *create_net_and_params())
    # %timeit profile_action_selection(no_static_argnums_egreedy, 100, *create_net_and_params())
    print(profile_action_selection(egreedy_jitted, 100, *create_net_and_params()))
    print(
        profile_action_selection(
            egreedy_jitted_no_static, 100, *create_net_and_params()
        )
    )


def mask_v_estimates(v_estimates, *_, **__):
    return v_estimates


def mask_q_estimates(q_estimates, *args, **_):
    """
    Given Q-values (a matrix of shape (replayed_states, n_actions)),
    extract the Q-values of the action chosen for replay and indexed
    by `replay_actions`.
    """
    replay_chosen_actions = args[0]
    return jax.vmap(lambda x, y: x[y])(q_estimates, replay_chosen_actions)


def train_module(
    net: nn.Module,
    params: FrozenDict,
    td_errors: jnp.DeviceArray,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: callable,
    states: np.ndarray,
    callback: callable,
    *args,
    **kwargs
) -> [Tuple[optax.OptState, FrozenDict, float]]:
    print(net)
    print(loss_fn)
    print(callback)

    # evaluation and loss function
    def estimate_states(params):
        estimates = jax.vmap(lambda state: net.apply(params, state))(states)
        estimates = callback(estimates.squeeze(), *args, **kwargs)
        return jnp.mean(jax.vmap(loss_fn)(td_errors, estimates))

    # optimize the network, taking the gradient of the loss function
    grad_fn = jax.value_and_grad(estimate_states)
    loss, grad = grad_fn(params)
    updates, opt_state = optim.update(grad, opt_state, loss)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


train_module_jit = jax.jit(train_module, static_argnums=(0, 3, 5, 7))

# constants
cartpole_state_shape = (4, 1)
cartpole_stack_size = 1
cartpole_num_actions = 2


# create dummy dqv agent with offline experience,
# easier to profile the desired functions, no interaction
# with env needed
def offline_exp_data():
    checkpoints_dir = "/home/xqz-u/uni/fourthYear/bsc-thesis/dopamine/thesis/tests/test_compare_dqv_dqn/dqn_data/checkpoints"
    checkpoints_iters = [356, 357]
    return exp_data.ExperimentData(
        seed=0,
        stack_size=cartpole_stack_size,
        batch_size=128,
        checkpoint_dir=checkpoints_dir,
        checkpoint_iterations=checkpoints_iters,
    )


def make_offline_dqv():
    return dqv.JaxDQVAgent(
        state_shape=cartpole_state_shape,
        num_actions=cartpole_num_actions,
        exp_data=offline_exp_data(),
    )


def profile_train_module(iters, agent):
    v_losses, q_losses = [], []
    for _ in range(iters):
        replay_elements = agent.sample_memory()
        td_error = dqv.dqv_td_error_jit(
            agent.V_network,
            agent.V_target,
            replay_elements["next_state"],
            replay_elements["reward"],
            replay_elements["terminal"],
            agent.exp_data.gamma,
        )
        agent.V_optim_state, agent.V_online, v_loss = train_module_jit(
            agent.V_network,
            agent.V_online,
            td_error,
            agent.optimizer,
            agent.V_optim_state,
            agent.exp_data.loss_fn,
            replay_elements["state"],
            mask_v_estimates,
        )
        agent.Q_optim_state, agent.Q_online, q_loss = train_module_jit(
            agent.Q_network,
            agent.Q_online,
            td_error,
            agent.optimizer,
            agent.Q_optim_state,
            agent.exp_data.loss_fn,
            replay_elements["state"],
            mask_q_estimates,
            replay_elements["action"],
        )
        v_losses.append(v_loss)
        q_losses.append(q_loss)
    return v_losses, q_losses


def test_train_module():
    print(profile_train_module(100, make_offline_dqv()))
