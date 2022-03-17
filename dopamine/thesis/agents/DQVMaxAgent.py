import functools as ft
from typing import Dict, Tuple

import attr
import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees
from thesis.agents import agent_utils
from thesis.agents.Agent import Agent


# TODO organize so that there is no need to pass both the
# NetworkOptimWrap and the FrozenDict parameters...
@ft.partial(jax.jit, static_argnums=(0))
def train_v_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    train_params: FrozenDict,
    states: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[custom_pytrees.NetworkOptimWrap, FrozenDict, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_params, td_targets)
    train_params, net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, train_params, net_optim.optim_state
    )
    return net_optim, train_params, loss


@ft.partial(jax.jit, static_argnums=(0))
def train_q_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    train_params: FrozenDict,
    states: np.ndarray,
    actions: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[
    custom_pytrees.NetworkOptimWrap, FrozenDict, jnp.DeviceArray, jnp.DeviceArray
]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        estimates = jax.vmap(lambda x, y: x[y])(estimates, actions)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates)), estimates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, q_estimates), grads = grad_fn(train_params, td_targets)
    train_params, net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, train_params, net_optim.optim_state
    )
    return net_optim, train_params, loss, q_estimates


@attr.s(auto_attribs=True)
class DQVMaxAgent(Agent):
    @property
    def model_names(self) -> Tuple[str]:
        return ("vfunc", "qfunc")

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [1, self.num_actions])
        # initialize target q network weights with online ones
        qfunc_params = self.models["qfunc"].params
        self.models["qfunc"].params = {"online": qfunc_params, "target": qfunc_params}

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(
            obs, self.models["qfunc"].net, self.models["qfunc"].params["online"]
        )

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        v_loss = self.train_v(replay_elts)
        q_loss, q_estimates = self.train_q(replay_elts)
        return {"loss": (v_loss, q_loss), "q_estimates": q_estimates}

    def train_v(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
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
            self.gamma,
            self.models["vfunc"],
            self.models["vfunc"].params,
            replay_elts["state"],
            v_td_targets,
        )
        return v_loss

    def train_q(self, replay_elts: Dict[str, np.ndarray]) -> Tuple[jnp.DeviceArray]:
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
            q_estimates,
        ) = train_q_net(
            self.gamma,
            self.models["qfunc"],
            self.models["qfunc"].params["online"],
            replay_elts["state"],
            replay_elts["action"],
            q_td_targets,
        )
        return q_loss, q_estimates

    def sync_weights(self):
        self.models["qfunc"].params["target"] = self.models["qfunc"].params["online"]
