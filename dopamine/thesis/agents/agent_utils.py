from typing import Dict, Tuple, Union

import attr
import optax
from dopamine.replay_memory import circular_replay_buffer
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from thesis import custom_pytrees, networks, offline_circular_replay_buffer


def build_net(
    out_dim: int,
    inp_shape: Tuple[int],
    key: custom_pytrees.PRNGKeyWrap,
    class_: nn.Module = networks.mlp,
    **kwargs,
) -> Tuple[nn.Module, FrozenDict]:
    net = class_(output_dim=out_dim, **kwargs)
    params = net.init(next(key), jnp.ones(inp_shape))
    return net, params


def build_optim(
    params: FrozenDict, class_: optax.GradientTransformation = optax.sgd, **kwargs
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optim = class_(**kwargs)
    optim_state = optim.init(params)
    return optim, optim_state


def attr_fields_d(attr_class) -> dict:
    return {
        field.name: getattr(attr_class, field.name)
        for field in attr.fields(type(attr_class))
    }


def sample_replay_buffer(
    memory: Union[
        circular_replay_buffer.OutOfGraphReplayBuffer,
        offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer,
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
