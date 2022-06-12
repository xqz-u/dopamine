import functools as ft
from typing import Callable, Dict, Tuple, Union

import jax
import numpy as np
import optax
from flax import linen as nn
from flax.core import frozen_dict
from jax import numpy as jnp
from thesis import custom_pytrees, memory, networks, types


def sample_replay_buffer(
    memory: Union[
        memory.OutOfGraphReplayBuffer,
        memory.OfflineOutOfGraphReplayBuffer,
    ],
    batch_size: int = None,
    indices: int = None,
) -> Dict[str, jnp.ndarray]:
    return dict(
        zip(
            [el.name for el in memory.get_transition_elements(batch_size=batch_size)],
            memory.sample_transition_batch(batch_size=batch_size, indices=indices),
        )
    )


def batch_net_eval(
    model_call: types.ModuleCall,
    params: frozen_dict.FrozenDict,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(lambda x: model_call(params, x))(inputs).squeeze()


# NOTE see
# https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html?highlight=TD(02#stopping-gradients
# as to why we use this construct here; the reson resides in the TD(0)
# update rule
def td_loss(
    discount: float,
    target_estimates: jnp.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
) -> jnp.ndarray:
    return jax.lax.stop_gradient(
        rewards + discount * target_estimates * (1.0 - terminals)
    )


@ft.partial(jax.jit, static_argnums=(0,))
def apply_td_loss(
    model_call: types.ModuleCall,
    params: frozen_dict.FrozenDict,
    experience_batch: Dict[str, np.ndarray],
    gamma: float,
) -> jnp.ndarray:
    return td_loss(
        gamma,
        model_call(params, experience_batch["next_state"]),
        experience_batch["reward"],
        experience_batch["terminal"],
    )


def build_models(models_tree: types.ModelDef) -> nn.Module:
    return models_tree[0](
        **{
            k: build_models(v)
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict)
            else v
            for k, v in models_tree[1].items()
        }
    )


# NOTE assumes that each head's name ends with its index, e.g.
# a 2-headed ensemble of a 2-layers MLP has structure:
# {params: {head_0: {Dense_0: {...}, Dense_1: ...}}, {head_1: {...}}}
# when training only the first head, this results in:
# {params: {head_0: trainable, head_1: zero}}
def mark_trainable_params(
    head_idx: int,
    params: frozen_dict.FrozenDict,
    pos_mark: str = "trainable",
    neg_mark: str = "zero",
) -> frozen_dict.FrozenDict:
    return frozen_dict.freeze(
        {
            "params": {
                h: pos_mark if h.endswith(f"_{head_idx}") else neg_mark
                for h in list(jax.tree_map(jnp.shape, params)["params"].keys())
            }
        }
    )


def build_TS(
    learner_def: types.ModelTSDef,
    rng: custom_pytrees.PRNGKeyWrap,
    input_shape: Tuple[int, ...],
    s_t_fn_def: types.ModuleCall,
    s_tp1_fn_def: types.ModuleCall,
    has_target_model: bool,
) -> custom_pytrees.ValueBasedTS:
    params = learner_def[0].init(next(rng), jnp.zeros(input_shape))
    return custom_pytrees.ValueBasedTS.create(
        apply_fn=s_t_fn_def,
        s_tp1_fn=s_tp1_fn_def,
        params=params,
        target_params=params if has_target_model else None,
        tx=learner_def[1],
        loss_metric=learner_def[2],
    )


def build_TS_ensemble(
    learner_def: types.ModelTSDef,
    rng: custom_pytrees.PRNGKeyWrap,
    input_shape: Tuple[int, ...],
    s_t_fn_def: Callable[..., types.ModuleCall],
    s_tp1_fn_def: types.ModuleCall,
    has_target_model: bool,
) -> custom_pytrees.ValueBasedTSEnsemble:
    model, optimizer, loss_metric = learner_def
    assert isinstance(model, networks.EnsembledNet)
    params = model.init(next(rng), jnp.zeros(input_shape))
    zero_optimizer = optax.set_to_zero()
    return custom_pytrees.ValueBasedTSEnsemble(
        tuple(
            custom_pytrees.ValueBasedTS.create(
                apply_fn=ft.partial(s_t_fn_def, i),
                s_tp1_fn=s_tp1_fn_def,
                params=params,
                target_params=params if has_target_model else None,
                tx=optax.multi_transform(
                    {"trainable": optimizer, "zero": zero_optimizer},
                    mark_trainable_params(i, params),
                ),
                loss_metric=loss_metric,
            )
            for i in range(model.n_heads)
        )
    )


# save the maximum Q-value for the first state of each episode
def t0_max_q_callback(
    episode_dict: types.MetricsDict, policy_eval_dict: types.MetricsDict
) -> types.MetricsDict:
    if episode_dict["steps"] == 0:
        episode_dict["max_q_s0"] = policy_eval_dict["max_q"]
    return episode_dict
