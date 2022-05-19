from typing import Dict, Tuple

import attr
import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, utils
from thesis.agents import agent_utils
from thesis.agents.Agent import Agent


# NOTE both functions receive a train_params argument to specify which
# set of function approximator parameters should be updated, e.g. DQV
# uses 2 sets of weights to approximate the V function while DQVMax and
# DQN do the same on the Q function
@jax.jit
def train_v_net(
    net_optim: custom_pytrees.NetworkOptimWrap,
    train_params: FrozenDict,
    states: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_params, td_targets)
    train_params, net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, train_params, net_optim.optim_state
    )
    return net_optim, train_params, loss


@jax.jit
def train_q_net(
    net_optim: custom_pytrees.NetworkOptimWrap,
    train_params: FrozenDict,
    states: np.ndarray,
    actions: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        estimates = jax.vmap(lambda x, y: x[y])(estimates, actions)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_params, td_targets)
    train_params, net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, train_params, net_optim.optim_state
    )
    return net_optim, train_params, loss


def default_sync_weights(self):
    self.models["qfunc"].params["target"] = self.models["qfunc"].params["online"]


def default_train_v(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
    v_td_targets = agent_utils.td_error(
        self.gamma,
        agent_utils.batch_net_eval(
            self.models["qfunc"].net,
            self.models["qfunc"].params["target"],
            replay_elts["next_state"],
        ).max(1),
        replay_elts["reward"],
        replay_elts["terminal"],
    )
    self.models["vfunc"], self.models["vfunc"].params, v_loss = train_v_net(
        self.models["vfunc"],
        self.models["vfunc"].params,
        replay_elts["state"],
        v_td_targets,
    )
    return v_loss


def default_train_q(self, replay_elts: Dict[str, np.ndarray]) -> Tuple[jnp.DeviceArray]:
    q_td_targets = agent_utils.td_error(
        self.gamma,
        agent_utils.batch_net_eval(
            self.models["vfunc"].net,
            self.models["vfunc"].params,
            replay_elts["next_state"],
        ),
        replay_elts["reward"],
        replay_elts["terminal"],
    )
    (
        self.models["qfunc"],
        self.models["qfunc"].params["online"],
        q_loss,
    ) = train_q_net(
        self.models["qfunc"],
        self.models["qfunc"].params["online"],
        replay_elts["state"],
        replay_elts["action"],
        q_td_targets,
    )
    return q_loss


# TODO base class for DQV-fam for parametrized train_q/v_func,
# so that attr_method_binder can become a decorator inside that class?
# and to have training routines for single head/multiple
# heads/generally different functions, which only need to implement
# the same interface, share the (possibly) similar structures
@attr.s(auto_attribs=True)
class DQVMaxAgent(Agent):
    train_v_func: callable = attr.ib(
        default=default_train_v, validator=utils.attr_method_binder
    )
    train_q_func: callable = attr.ib(
        default=default_train_q, validator=utils.attr_method_binder
    )
    sync_weights_func: callable = attr.ib(
        default=default_sync_weights, validator=utils.attr_method_binder
    )

    @property
    def model_names(self) -> Tuple[str]:
        return ("vfunc", "qfunc")

    @property
    def repr_name(self) -> str:
        return "DQVMax"

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [1, self.num_actions])
        # initialize target q network weights with online ones
        qfunc_params = self.models["qfunc"].params
        self.models["qfunc"].params = {"online": qfunc_params, "target": qfunc_params}

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, ...]:
        return self._select_action(
            obs, self.models["qfunc"].net, self.models["qfunc"].params["online"]
        )

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        return {
            "loss": (self.train_v_func(replay_elts), self.train_q_func(replay_elts))
        }

    def sync_weights(self):
        self.sync_weights_func()
