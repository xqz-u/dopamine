import functools as ft
from typing import Dict, List, Tuple

import attr
import jax
import numpy as np
from jax import numpy as jnp
from thesis import custom_pytrees
from thesis.agents import agent_base, agent_utils


@ft.partial(jax.jit, static_argnums=(0))
def train_q_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    states: np.ndarray,
    actions: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        estimates = jax.vmap(lambda x, y: x[y])(estimates, actions)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(net_optim.params["online"], td_targets)
    net_optim.params["online"], net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, net_optim.params["online"], net_optim.optim_state
    )
    return net_optim, loss


@ft.partial(jax.jit, static_argnums=(0))
def train_v_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    states: np.ndarray,
    td_targets: jnp.DeviceArray,
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = agent_utils.batch_net_eval(net_optim.net, params, states)
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(net_optim.params, td_targets)
    net_optim.params, net_optim.optim_state = agent_utils.optimize(
        net_optim.optim, grads, net_optim.params, net_optim.optim_state
    )
    return net_optim, loss


# TODO agnostic on networks names? and then set them as derived class
# @property es? or just do the latter without the first?
@attr.s(auto_attribs=True)
class DQVMaxAgent(agent_base.Agent):
    def build_networks_and_optimizers(self):
        self.model_names = ["qnet", "vnet"]
        out_dims = [self.num_actions, 1]
        self._build_networks_and_optimizers(self.model_names, out_dims)
        # initialize target q network weights with online ones
        qnet_params = self.models["qnet"].params
        self.models["qnet"].params = {
            "online": qnet_params,
            "target": qnet_params,
        }

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(
            obs,
            self.models["qnet"].net,
            self.models["qnet"].params["online"],
        )

    def train(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
        return jnp.array(
            (self.train_v(replay_elts), self.train_q(replay_elts))
        ).reshape((2, 1))

    def train_v(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
        v_td_targets = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["qnet"].net,
                self.models["qnet"].params["target"],
                replay_elts["next_state"],
            ).max(1),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        self.models["vnet"], v_loss = train_v_net(
            self.gamma, self.models["vnet"], replay_elts["state"], v_td_targets
        )
        return v_loss

    def train_q(self, replay_elts: Dict[str, np.ndarray]) -> jnp.DeviceArray:
        q_td_targets = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["vnet"].net,
                self.models["vnet"].params,
                replay_elts["next_state"],
            ),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        self.models["qnet"], q_loss = train_q_net(
            self.gamma,
            self.models["qnet"],
            replay_elts["state"],
            replay_elts["action"],
            q_td_targets,
        )
        return q_loss

    def sync_weights(self):
        self.models["qnet"].params["target"] = self.models["qnet"].params["online"]

    # TODO modify unbundle and what else the runner uses to allow for
    # nested dictionaries, also because I do not think the unbundle
    # agent_base method can deal with them correctly rn
    def bundle_and_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> dict:
        q_params = self.models["qnet"].params
        q_dict = {
            "q_params_online": q_params["online"],
            "q_params_target": q_params["target"],
            "q_opt_state": self.models["qnet"].opt_state,
        }
        v_dict = {
            "v_params": self.models["vnet"].params,
            "v_opt_state": self.models["vnet"].opt_state,
        }
        return {
            **q_dict,
            **v_dict,
            **super().bundle_and_checkpoint(checkpoint_dir, iteration_number),
        }

    def unbundle(self):
        # do something here
        pass
