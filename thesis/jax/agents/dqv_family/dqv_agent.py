import functools as ft
from typing import Tuple

import attr
import gin
import jax
import numpy as onp
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from thesis import utils as u
from thesis.jax.agents.dqv_family import dqv_base

# NOTE many functions and class methods are called with no args when they are
# actually required; args passing is intended to be done with gin. In any case,
# ideally one would still pass the required parameters, and not pass values for
# those args that already have sensible defaults plus can be configured with gin


def mask_v_estimates(v_estimates: jnp.DeviceArray, *_, **__) -> jnp.DeviceArray:
    return v_estimates


def mask_q_estimates(q_estimates: jnp.DeviceArray, *args, **_) -> jnp.DeviceArray:
    """
    Given Q-values (a matrix of shape (replayed_states, n_actions)),
    extract the Q-values of the action chosen for replay and indexed
    by `replay_actions`.
    """
    replay_chosen_actions = args[0]
    return jax.vmap(lambda x, y: x[y])(q_estimates, replay_chosen_actions)


@ft.partial(jax.jit, static_argnums=(0, 3, 5, 7))
def train_module(
    net: nn.Module,
    params: FrozenDict,
    td_errors: jnp.DeviceArray,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: callable,
    states: onp.ndarray,
    mask: callable,
    *args,
    **kwargs
) -> [Tuple[optax.OptState, FrozenDict, float]]:
    # evaluation and loss function
    def estimate_states(params):
        estimates = jax.vmap(lambda state: net.apply(params, state))(states)
        estimates = mask(estimates.squeeze(), *args, **kwargs)
        return jnp.mean(jax.vmap(loss_fn)(td_errors, estimates))

    # optimize the network, taking the gradient of the loss function
    grad_fn = jax.value_and_grad(estimate_states)
    loss, grad = grad_fn(params)
    updates, opt_state = optim.update(grad, opt_state, loss)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss


# NOTE the update in the paper by Matthia computes TD errors discriminating
# against terminal _next_states_, whereas here I am using terminal _states_, but
# it should be the same
@ft.partial(jax.jit, static_argnums=(0, 5))
def dqv_td_error(
    vnet: nn.Module, target_params: FrozenDict, next_states, rewards, terminals, gamma
) -> jnp.DeviceArray:
    v_values = jax.vmap(lambda state: vnet.apply(target_params, state))(next_states)
    # needed, vmap might create a column vector to vectorize operation on states
    v_values = v_values.squeeze()
    return rewards + gamma * v_values * (1 - terminals)


# TODO offline agent version
# TODO eval mode
# TODO unbundle method to restart from checkpoint
# TODO check if an `update_period` for memory sampling gives better performance
@gin.configurable
@attr.s(auto_attribs=True)
class JaxDQVAgent(dqv_base.DQV):
    Q_online: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()
    V_target: FrozenDict = FrozenDict()

    # NOTE other parameters of the networks should be already bound
    def build_networks(self):
        self.rng, rng0, rng1 = u.force_devicearray_split(self.rng, 3)
        self.V_network = self.V_network(output_dim=1)
        self.Q_network = self.Q_network(output_dim=self.num_actions)
        self.V_online = self.V_network.init(rng0, self.state)
        self.V_target = self.V_online
        self.Q_online = self.Q_network.init(rng1, self.state)

    # 0. if running number of interactions with env is > than N: train phase
    # 1. sample mini-batch of size Z of transitions from replay memory
    # 2. compute TD-error
    # 3. compute loss on TD-error and train the NNs
    # 4. sync weights between online and target networks with frequency X
    def _train_step(self):
        if self.memory.add_count > self.exp_data.min_replay_history:
            replay_elements = self.sample_memory()
            td_error = dqv_td_error(
                self.V_network,
                self.V_target,
                replay_elements["next_state"],
                replay_elements["reward"],
                replay_elements["terminal"],
                self.exp_data.gamma,
            )
            self.V_optim_state, self.V_online, v_loss = train_module(
                self.V_network,
                self.V_online,
                td_error,
                self.optimizer,
                self.V_optim_state,
                self.exp_data.loss_fn,
                replay_elements["state"],
                mask_v_estimates,
            )
            self.Q_optim_state, self.Q_online, q_loss = train_module(
                self.Q_network,
                self.Q_online,
                td_error,
                self.optimizer,
                self.Q_optim_state,
                self.exp_data.loss_fn,
                replay_elements["state"],
                mask_q_estimates,
                replay_elements["action"],
            )
            self.save_summaries(v_loss, q_loss)
            if self.training_steps % self.exp_data.target_update_period == 0:
                self.V_target = self.V_online
        self.training_steps += 1

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        return super().bundle_and_checkpoint(
            checkpoint_dir,
            iteration_number,
            **{
                "Q_online": self.Q_online,
                "V_online": self.V_online,
                "V_target": self.V_target,
            },
        )

    @property
    def networks_shape(self):
        return dict(
            map(
                lambda attr: (
                    attr,
                    jax.tree_map(jnp.shape, getattr(self, attr)).unfreeze(),
                ),
                ["Q_online", "V_online", "V_target"],
            )
        )
