import functools as ft
from typing import Callable, List, NamedTuple, Tuple

import jax
import optax
from dopamine.jax import losses
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from jax import numpy as jnp
from jax import random as jrand
from thesis import custom_pytrees, networks
from thesis.agent import utils as agent_utils


@jax.jit
def train_Q(
    tr_state: custom_pytrees.ValueBasedTS,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    td_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, custom_pytrees.ValueBasedTS]:
    def loss_fn(params, targets):
        qs = jax.vmap(lambda x: tr_state.apply_fn(params, x))(states).squeeze()
        played_qs = jax.vmap(lambda q, a: q[a])(qs, actions)

        return jnp.mean(jax.vmap(tr_state.loss_metric)(targets, played_qs))

    loss, grads = jax.value_and_grad(loss_fn)(tr_state.params, td_targets)
    return loss, tr_state.apply_gradients(grads=grads)


class FinalHead(nn.Module):
    features: int

    @nn.compact
    def __call__(
        self,
        backbone_params: FrozenDict,
        x: jnp.ndarray,
        backbone_module: nn.Module = None,
    ) -> jnp.ndarray:
        x = backbone_module.apply(backbone_params, x)
        x = nn.Dense(self.features)(x)
        return x


obs_shape = (4, 1)
n_heads = 2
gamma = 0.99

rng = custom_pytrees.PRNGKeyWrap(42)

# NOTE features is whatever for the body? like an additional hidden
# layer?
shared_mlp = networks.MLP(features=3, hiddens=(4,))
shared_mlp_params = shared_mlp.init(next(rng), jnp.zeros(obs_shape))

head_model = FinalHead(features=2)
heads_params = [
    head_model.init(
        next(rng), shared_mlp_params, jnp.zeros(obs_shape), backbone_module=shared_mlp
    )
    for _ in range(n_heads)
]

heads_full_params = list(zip(heads_params, [shared_mlp_params] * n_heads))

apply_fn = lambda head_and_body_params, x: ft.partial(
    head_model.apply, backbone_module=shared_mlp
)(*head_and_body_params, x)
# apply_fn_partial = jax.tree_util.Partial(head_model.apply, shared_mlp)
tss = [
    custom_pytrees.ValueBasedTS.create(
        params=head_ps,
        apply_fn=apply_fn,
        # s_tp1_fn=apply_fn_partial,
        s_tp1_fn=None,  # is this really needed at all??
        tx=optax.adam(**{"learning_rate": 0.001, "eps": 3.125e-4}),
        target_params=head_ps,
        loss_metric=losses.mse_loss,
    )
    for head_ps in heads_full_params
]


# generate some fake data
batch_size = 8
batch = {
    "state": jrand.uniform(next(rng), (batch_size,) + obs_shape),
    "next_state": jrand.uniform(next(rng), (batch_size,) + obs_shape),
    "reward": jrand.uniform(next(rng), (batch_size,)),
    "action": jrand.randint(next(rng), (batch_size,), 0, 2),
    "terminal": jrand.randint(next(rng), (batch_size,), 0, 2),
}

# say head 0 will train this episode...
head_i = 0
episode_head_ts = tss[head_i]
# TODO named tuple to have shared and head params easily recognizable!
# td_targets = agent_utils.apply_td_loss(
#     episode_head_ts.apply_fn, episode_head_ts.target_params[0], batch, gamma
# )
td_targets = agent_utils.apply_td_loss(
    lambda params, xs: jax.vmap(lambda x: episode_head_ts.apply_fn(params, x))(xs)
    .squeeze()
    .max(1),
    episode_head_ts.target_params,
    batch,
    gamma,
)
loss, tss[head_i] = train_Q(
    episode_head_ts, batch["state"], batch["action"], td_targets
)


class QParams(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class ValueBasedTS(train_state.TrainState):
    loss_metric: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = struct.field(
        pytree_node=False
    )
    backbone_params: FrozenDict
    heads_params = List[QParams]
