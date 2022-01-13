import functools as ft
from typing import Dict, Tuple

import attr
import jax
import numpy as np
import optax
from dopamine.jax import losses
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees
from thesis.agents import agent_base


@ft.partial(jax.jit, static_argnums=(0))
def train_q_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    replay_elts: Dict[str, np.ndarray],
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = jax.vmap(lambda s: net_optim.net.apply(params, s))(
            replay_elts["state"]
        )
        estimates = jax.vmap(lambda x, y: x[y])(estimates, replay_elts["action"])
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    td_targets = q_td_error(
        discount, net_optim.net, net_optim.params["target"], replay_elts
    )
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(net_optim.params["online"], td_targets)
    net_optim.params["online"], net_optim.optim_state = optimize(
        net_optim.optim, grads, net_optim.params["online"], net_optim.optim_state
    )
    return net_optim, loss


@ft.partial(jax.jit, static_argnums=(0))
def train_v_net(
    discount: float,
    net_optim: custom_pytrees.NetworkOptimWrap,
    replay_elts: Dict[str, np.ndarray],
) -> Tuple[custom_pytrees.NetworkOptimWrap, jnp.DeviceArray]:
    def loss_fn(params, targets) -> jnp.DeviceArray:
        estimates = jax.vmap(lambda s: net_optim.net.apply(params, s))(
            replay_elts["state"]
        )
        return jnp.mean(jax.vmap(net_optim.loss_metric)(targets, estimates))

    td_targets = v_td_error(discount, net_optim.net, net_optim.params, replay_elts)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(net_optim.params, td_targets)
    net_optim.params, net_optim.optim_state = optimize(
        net_optim.optim, grads, net_optim.params, net_optim.optim_state
    )
    return net_optim, loss


def q_td_error(
    discount: float,
    net: nn.Module,
    params: FrozenDict,
    replay_elts: Dict[str, np.ndarray],
):
    target_estimates = jax.vmap(lambda s: net.apply(params, s))(
        replay_elts["next_state"]
    ).max(axis=1)
    return td_error(discount, target_estimates, replay_elts)


def v_td_error(
    discount: float,
    net: nn.Module,
    params: FrozenDict,
    replay_elts: Dict[str, np.ndarray],
) -> jnp.DeviceArray:
    target_estimates = jax.vmap(lambda s: net.apply(params, s))(
        replay_elts["next_state"]
    )
    return td_error(discount, target_estimates, replay_elts)


def td_error(
    discount: float,
    target_estimates: jnp.DeviceArray,
    replay_elts: Dict[str, np.ndarray],
) -> jnp.DeviceArray:
    return jax.lax.stop_gradient(
        replay_elts["reward"]
        + discount * target_estimates * (1.0 - replay_elts["terminal"])
    )


# NOTE idk if I can call this from a jitted function, first arg is not
# a valid pytree
def optimize(
    optim: optax.GradientTransformation,
    grads: FrozenDict,
    params: FrozenDict,
    optim_state: optax.OptState,
) -> Tuple[FrozenDict, optax.OptState]:
    updates, optim_state = optim.update(grads, optim_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, optim_state


# TODO agnostic on networks names? and then set them as derived class
# @property es? or just do the latter without the first?
@attr.s(auto_attribs=True)
class DQVMaxAgent(agent_base.Agent):
    def build_networks_and_optimizers(self):
        net_names, out_dims = ["qnet", "vnet"], [self.num_actions, 1]
        self._build_networks_and_optimizers(net_names, out_dims)
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

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        self.models["vnet"], v_loss = train_v_net(
            self.gamma,
            self.models["vnet"],
            replay_elts,
        )
        self.models["qnet"], q_loss = train_q_net(
            self.gamma,
            self.models["qnet"],
            replay_elts,
        )
        return {
            f"{n}_{self.models[n].loss_metric.__name__}": v
            for n, v in zip(["vnet", "qnet"], [v_loss, q_loss])
        }

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


import optax
from thesis import exploration, networks, offline_circular_replay_buffer

conf = {
    "nets": {
        "qnet": {
            "model": {
                "class_": networks.mlp,
                "hiddens": (5, 5),
            },
            "optim": {
                "class_": optax.sgd,
                "learning_rate": 0.09,
            },
            "loss": losses.mse_loss,
        },
        "vnet": {"optim": {"learning_rate": 0.05}},
    },
    "exploration": {
        "fn": exploration.egreedy_linear_decay,
        # "fn": exploration.egreedy,
    },
    # "memory": {
    #     "stack_size": 1,
    #     "replay_capacity": 50000,
    #     "batch_size": 128,
    # },
    "memory": {
        "class_": offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
        "stack_size": 1,
        "checkpoint_dir": "/home/xqz-u/uni/dopamine/resources/data/JaxDQNAgent_CartPole-v0_ClassicControlDQNNetwork_ref_1_1641323676/checkpoints",
        "iterations": [499],
    },
    "experiment": {
        "seed": 4,
        "steps": 1000,
        "iterations": 500,
        "redundancy": 5,
        "env": "CartPole-v0",
    },
    "agent": {
        "class_": DQVMaxAgent,
        "net_sync_freq": 2000,
        "min_replay_history": 5000,
        # "train_freq": 4,
        # "gamma": 0.99,
    },
    # "logs": {},
}

import gym


def make_agent(config: dict, env: gym.Env):
    observation_shape = env.observation_space.shape + (1,)
    return config["agent"]["class_"](
        config,
        num_actions=env.action_space.n,
        observation_shape=observation_shape,
        observation_dtype=env.observation_space.dtype,
    )


env = gym.make("CartPole-v0")
max_ag = make_agent(conf, env)

obs = env.reset()

action = max_ag.select_action(obs)
obs, reward, done, _ = env.step(action)
losses = max_ag.learn(obs, reward, done)
# print(losses)


# replay_elts = max_ag.sample_memory()
# jax.make_jaxpr(train_v_net, static_argnums=(0))(
#     max_ag.gamma, max_ag.models["vnet"], replay_elts
# )
