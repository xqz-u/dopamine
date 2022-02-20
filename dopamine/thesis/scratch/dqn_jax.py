import functools as ft
import time
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import flax
import gin
import numpy as np
import optax
import tensorflow as tf
from dopamine.jax import losses, networks
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn

import jax
from jax import numpy as jnp
from jax import random as jrand
from thesis import utils

cartpole_min_vals = jnp.array(gin.query_parameter("jax_networks.CARTPOLE_MIN_VALS"))
cartpole_max_vals = jnp.array(gin.query_parameter("jax_networks.CARTPOLE_MAX_VALS"))


# NOTE move preproc outside of here
class Sequential(nn.Module):
    layers: Sequence[nn.Module]
    _min_vals: jnp.DeviceArray = cartpole_min_vals
    _max_vals: jnp.DeviceArray = cartpole_max_vals

    def __call__(self, x):
        x -= self._min_vals
        x /= self._max_vals - self._min_vals
        x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        for layer in self.layers:
            x = layer(x)
        return x

    def __hash__(self):
        return ft.reduce(lambda acc, x: hash(hash(x) + acc), self.layers, 0)


def mlp(
    outputs: Sequence[int],
    kernel_initializer: callable = nn.initializers.xavier_uniform(),
    activation_fn: callable = nn.relu,
) -> nn.Module:
    layers_and_activations = [
        v
        for tup in zip(
            [
                nn.Dense(features=out_feat, kernel_init=kernel_initializer)
                for out_feat in outputs
            ],
            [activation_fn] * len(outputs),
        )
        for v in tup
    ]
    return Sequential(layers_and_activations[:-1])


net_eval = lambda net, params, inp: net.apply(params, inp.squeeze())
batch_net_eval = lambda net, params, inputs: jax.vmap(
    net_eval, in_axes=(None, None, 0)
)(net, params, inputs)


@ft.partial(jax.jit, static_argnums=(3, 4, 5))
def egreedy_selection(
    state: np.ndarray,
    params: flax.core.frozen_dict.FrozenDict,
    rng: jrand.PRNGKey,
    num_actions: int,
    epsilon: float,
    q_net: nn.Module,
) -> Tuple[jrand.PRNGKey, jnp.DeviceArray]:
    rng, k0, k1 = jrand.split(rng, 3)
    return rng, jnp.where(
        jrand.uniform(k0) <= epsilon,
        jrand.randint(k1, (), 0, num_actions),
        jnp.argmax(net_eval(q_net, params, state)),
    )


def td_targets(
    net: nn.module,
    discount: float,
    next_states,
    rewards,
    terminals,
    params: flax.core.frozen_dict.FrozenDict,
) -> jnp.DeviceArray:
    return jax.lax.stop_gradient(
        rewards
        + discount * batch_net_eval(net, params, next_states).max(1) * (1 - terminals)
    )


@ft.partial(jax.jit, static_argnums=(0, 1, 2))
def optimize(
    net: nn.module,
    optim: optax.GradientTransformation,
    discount: float,
    states,
    actions,
    next_states,
    rewards,
    terminals,
    params: flax.core.frozen_dict.FrozenDict,
    target_params: flax.core.frozen_dict.FrozenDict,
    optim_state: optax.OptState,
) -> tuple[optax.OptState, flax.core.frozen_dict.FrozenDict, jnp.DeviceArray]:
    def loss_fn(params):
        bellman_targets = td_targets(
            net, discount, next_states, rewards, terminals, target_params
        )
        q_vs = batch_net_eval(net, params, states)
        q_replay_act = jax.vmap(lambda x, y: x[y])(q_vs, actions)
        return jnp.mean(jax.vmap(losses.huber_loss)(bellman_targets, q_replay_act))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, optim_state = optim.update(grads, optim_state, params=params)
    return optim_state, optax.apply_updates(params, updates), loss


@dataclass
class DQNAgent:
    num_actions: int
    observation_shape: tuple
    seed: int = None
    stack_size: int = 1
    discount: float = 0.99
    epsilon: float = 0.01
    min_replay_history: int = 500
    target_update_p: int = 100
    train_p: int = 4
    rng: jrand.PRNGKey = None
    state: jnp.DeviceArray = None
    action: np.ndarray = None
    memory: circular_replay_buffer.OutOfGraphReplayBuffer = None
    q_net: nn.Module = None
    q_online_params: flax.core.frozen_dict.FrozenDict = None
    q_target_params: flax.core.frozen_dict.FrozenDict = None
    optim: optax.GradientTransformation = None
    optim_state: optax.OptState = None
    training_steps: int = 0
    summary_writer: tf.summary.SummaryWriter = None
    summary_writing_freq: int = 500
    _avg_loss: jnp.DeviceArray = jnp.array(0.0)

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(time.time() * 1e6)
        self.rng = jrand.PRNGKey(self.seed)
        self.state = np.zeros(self.state_shape)
        self.init_networks()
        self.init_optimizers()

    @property
    def state_shape(self) -> tuple:
        # return self.observation_shape + (self.stack_size,)
        return self.observation_shape

    @property
    def can_summarise(self):
        return (
            self.summary_writer is not None
            and self.training_steps > 0
            and self.training_steps % self.summary_writing_freq == 0
        )

    def init_networks(self):
        self.rng, k0 = jrand.split(self.rng)
        self.q_online_params = self.q_net.init(k0, self.state)
        self.q_target_params = self.q_online_params

    def init_optimizers(self):
        self.optim_state = self.optim.init(self.q_online_params)

    def sample_memory(self, batch_size=None, indices=None) -> Dict[str, np.ndarray]:
        return dict(
            zip(
                [
                    el.name
                    for el in self.memory.get_transition_elements(batch_size=batch_size)
                ],
                self.memory.sample_transition_batch(
                    batch_size=batch_size, indices=indices
                ),
            )
        )

    def record_trajectory(self, reward: float, terminal: bool):
        self.memory.add(self.state, self.action, reward, terminal)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        self.state = np.reshape(obs, self.state_shape)
        self.rng, self.action = egreedy_selection(
            self.state,
            self.q_online_params,
            self.rng,
            self.num_actions,
            self.epsilon,
            self.q_net,
        )
        self.action = np.array(self.action)
        return self.action

    def learn(self, _, reward: float, terminal: bool):
        self.record_trajectory(reward, terminal)
        if terminal:
            return
        if self.memory.add_count >= self.min_replay_history:
            if self.training_steps % self.train_p == 0:
                replay_elts = self.sample_memory()
                self.optim_state, self.q_online_params, loss = optimize(
                    self.q_net,
                    self.optim,
                    self.discount,
                    replay_elts["state"],
                    replay_elts["action"],
                    replay_elts["next_state"],
                    replay_elts["reward"],
                    replay_elts["terminal"],
                    self.q_online_params,
                    self.q_target_params,
                    self.optim_state,
                )
                self._avg_loss = (self._avg_loss + loss) / 2
                if self.can_summarise:
                    utils.add_summary_v2(
                        self.summary_writer,
                        [
                            ["scalar", "HuberLoss", np.array(loss)],
                        ],
                        self.training_steps,
                    )
            if self.training_steps % self.target_update_p == 0:
                self.q_target_params = self.q_online_params
        self.training_steps += 1
