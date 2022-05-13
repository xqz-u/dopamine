import functools as ft
from typing import Dict, Tuple, Union

import jax
import numpy as np
import optax
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, networks
from thesis.memory import offline_memory, prio_offline_memory


def build_net(
    out_dim: int,
    example_inp: jnp.DeviceArray,
    key: custom_pytrees.PRNGKeyWrap,
    call_: nn.Module = networks.mlp,
    **kwargs,
) -> Tuple[nn.Module, FrozenDict, dict]:
    args = locals()
    net = call_(out_dim, **kwargs)
    params = net.init(next(key), example_inp)
    return net, params, {"call_": args["call_"], **args["kwargs"]}


def build_optim(
    params: FrozenDict, call_: optax.GradientTransformation = optax.sgd, **kwargs
) -> Tuple[optax.GradientTransformation, optax.OptState, dict]:
    args = locals()
    optim = call_(**kwargs)
    optim_state = optim.init(params)
    return optim, optim_state, {"call_": args["call_"], **args["kwargs"]}


def sample_replay_buffer(
    memory: Union[
        circular_replay_buffer.OutOfGraphReplayBuffer,
        offline_memory.OfflineOutOfGraphReplayBuffer,
        prio_offline_memory.PrioritizedOfflineOutOfGraphReplayBuffer,
    ],
    batch_size: int = None,
    indices: int = None,
) -> Dict[str, jnp.DeviceArray]:
    return dict(
        zip(
            [el.name for el in memory.get_transition_elements(batch_size=batch_size)],
            memory.sample_transition_batch(batch_size=batch_size, indices=indices),
        )
    )


@ft.partial(jax.jit, static_argnums=(0))
def batch_net_eval(
    net: nn.Module, params: FrozenDict, xs: jnp.DeviceArray
) -> jnp.DeviceArray:
    return jax.vmap(lambda s: net.apply(params, s))(xs).squeeze()


@ft.partial(jax.jit, static_argnums=(0))
def td_error(
    discount: float,
    target_estimates: jnp.DeviceArray,
    rewards: np.ndarray,
    terminals: np.ndarray,
) -> jnp.DeviceArray:
    return jax.lax.stop_gradient(
        rewards + discount * target_estimates * (1.0 - terminals)
    )


# NOTE called by jitted routines usually
def optimize(
    optim: optax.GradientTransformation,
    grads: FrozenDict,
    params: FrozenDict,
    optim_state: optax.OptState,
) -> Tuple[FrozenDict, optax.OptState]:
    updates, optim_state = optim.update(grads, optim_state, params=params)
    return optax.apply_updates(params, updates), optim_state
