import functools as ft
import pprint
from typing import Callable, Tuple

import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import configs, custom_pytrees, networks, types, utils
from thesis.agent import dqn
from thesis.agent import utils as agent_utils

# NOTE problem with this implementation is that


class EnsembledHead(nn.Module):
    features: int
    backbone_fn: Callable[..., nn.Module]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.features, name="head")(
            self.backbone_fn(name="shared_body")(x)
        )


# NOTE modifies learner_def.info, adding a string representation of the
# shared model which is passed as a partialled function;
def build_shared_TS_ensemble(
    learner_def: agent_utils.ModelDefStore,
    rng: custom_pytrees.PRNGKeyWrap,
    input_shape: Tuple[int, ...],
    s_t_fn_def: types.ModuleCall,
    # s_tp1_fn_def: types.ModuleCall,  # ????
    has_target_model: bool = True,
) -> custom_pytrees.ValueBasedTSEnsemble:
    def init_head() -> FrozenDict:
        head_params = learner_def.net.init(next(rng), example_input).unfreeze()
        head_params["params"]["shared_body"] = shared_params["params"]
        return FrozenDict(head_params)

    assert isinstance(learner_def.net, EnsembledHead) and "n_heads" in learner_def.info
    example_input = jnp.ones(obs_shape)
    shared_params = learner_def.net.backbone_fn().init(next(rng), example_input)
    learner_def.info["backbone_fn"] = str(learner_def.net.backbone_fn)
    return custom_pytrees.ValueBasedTSEnsemble(
        [
            custom_pytrees.ValueBasedTS.create(
                params=full_params,
                target_params=full_params if has_target_model else None,
                apply_fn=s_t_fn_def,
                s_tp1_fn=None,
                tx=learner_def.opt(**learner_def.opt_params),
                loss_metric=ft.partial(
                    learner_def.loss_fn, **learner_def.loss_fn_params
                ),
            )
            for full_params in [init_head() for _ in range(learner_def.info["n_heads"])]
        ]
    )


obs_shape = (4, 1)
n_heads = 2
gamma = 0.99

rng = custom_pytrees.PRNGKeyWrap(42)


# NOTE use a named backbone_fn for utils.reportable_config to at least
# get the name of the function that defins the backbone_model -
# jax.tree_map won't descend since a function is not a pytree
net_def = (
    EnsembledHead,
    {
        "features": 2,
        "backbone_fn":
        # (networks.MLP, {"features": 3, "hiddens": (4,)})
        (lambda: jax.tree_util.Partial(networks.MLP, features=3, hiddens=(4,)), {}),
    },
)
learner_def = agent_utils.ModelDefStore(
    net_def=net_def, **configs.make_adam_mse_def(), info={"n_heads": n_heads}
)
tss = build_shared_TS_ensemble(learner_def, rng, obs_shape, learner_def.net.apply)

pprint.pprint(
    utils.reportable_config(utils.config_collector(learner_def, "reportable"))
)

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
td_targets = agent_utils.apply_td_loss(
    lambda params, xs: jax.vmap(lambda x: episode_head_ts.apply_fn(params, x))(xs)
    .squeeze()
    .max(1),
    episode_head_ts.target_params,
    batch,
    gamma,
)
loss, tss[head_i] = dqn.train_Q(
    episode_head_ts, batch["state"], batch["action"], td_targets
)

# after the episode is finished, the backbone parameters get updated
# for all heads, so that a shared representation is learned
def propagate_head_representation(
    tss: custom_pytrees.ValueBasedTSEnsemble, trained_idx: int
) -> custom_pytrees.ValueBasedTSEnsemble:

    ...
