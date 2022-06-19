from typing import Any, Dict

import gym
import optax
from dopamine.discrete_domains import atari_lib
from dopamine.jax import losses

from thesis import constants, custom_pytrees, exploration, memory, networks, reporter
from thesis.agent import utils as agent_utils

# -------------------------------------------------------------
# env


# see dopamine.discrete_domains.atari_lib.create_atari_environment's
# docs for more comments; rewrote here to allow creation of environments
# under the ALE/ namespace too
def create_atari_environment(
    env_name: str, env_args: Dict[str, Any] = None
) -> atari_lib.AtariPreprocessing:
    return atari_lib.AtariPreprocessing(gym.make(env_name).env)


# -------------------------------------------------------------
# networks, loss metrics, optimizers definitions
# (enpoints return agent_utils.ModelDefStore)

make_mlp_def = lambda features, env_name, **kwargs: (
    networks.MLP,
    {
        "features": features,
        **constants.env_preproc_info.get(env_name, {}),
        **kwargs,
    },
)


make_ensemble_def = lambda n_heads, heads_model: (
    networks.EnsembledNet,
    {"n_heads": n_heads, "model": heads_model},
)


# like Dopamine's
make_adam_mse_def = lambda: {
    "opt": optax.adam,
    "opt_params": {
        "learning_rate": 0.001,
        "eps": 3.125e-4,
    },
    "loss_fn": losses.mse_loss,
}


adam_mse_mlp = lambda features, env_name, **mlp_kwargs: agent_utils.ModelDefStore(
    **{"net_def": make_mlp_def(features, env_name, **mlp_kwargs), **make_adam_mse_def()}
)


adam_mse_ensemble_mlp = (
    lambda n_heads, features, env_name, **mlp_kwargs: agent_utils.ModelDefStore(
        **{
            "net_def": make_ensemble_def(
                n_heads, make_mlp_def(features, env_name, **mlp_kwargs)
            ),
            **make_adam_mse_def(),
        }
    )
)


# NOTE pickle cannot serialize some elements, e.g. flax.linen.relu.
# one hack is to serialize such elements with cloudpickle, then to
# deserialize them in the spawned process; or pass to mp.Pool.map a
# function that returns the args you wanted to pass in.
# the solution would be to swap mp.Pool's serializer with cloudpickle
# one's, but I can't make it work

# NOTE these functions create all models with same kwargs; if different
# parameters are desired across models, e.g. a Q ensemble with different
# heads than the V one for DQVMax, create ad-hoc function which accepts
# env_name, out_dim, **kwargs; it can build upon some of these fns

# named functions accepted by base mp.Pool serializer
def dqn_model_maker(
    env_name: str, out_dim: int, **kwargs
) -> Dict[str, agent_utils.ModelDefStore]:
    return {"Q_model_def": adam_mse_mlp(out_dim, env_name, **kwargs)}


def dqvmax_model_maker(
    env_name: str, q_out_dim: int, **kwargs
) -> Dict[str, agent_utils.ModelDefStore]:
    return {
        **dqn_model_maker(env_name, q_out_dim, **kwargs),
        "V_model_def": adam_mse_mlp(1, env_name, **kwargs),
    }


# ensembles
def dqn_ensemble_model_maker(
    env_name: str, out_dim: int, heads: int, **kwargs
) -> Dict[str, agent_utils.ModelDefStore]:
    return {"Q_model_def": adam_mse_ensemble_mlp(heads, out_dim, env_name, **kwargs)}


def dqv_ensemble_model_maker(
    env_name: str, q_out_dim: int, heads: int, **kwargs
) -> Dict[str, agent_utils.ModelDefStore]:
    return {
        "V_model_def": adam_mse_ensemble_mlp(heads, 1, env_name, **kwargs),
        **dqn_model_maker(env_name, q_out_dim, **kwargs),
    }


def dqvmax_ensemble_model_maker(
    env_name: str, q_out_dim: int, heads: int, **kwargs
) -> Dict[str, agent_utils.ModelDefStore]:
    return {
        **dqn_ensemble_model_maker(env_name, q_out_dim, env_name, **kwargs),
        "V_model_def": adam_mse_ensemble_mlp(heads, 1, env_name, **kwargs),
    }


# -------------------------------------------------------------
# policy evaluators, default to Egreedy
make_explorer = lambda env, expl_class=exploration.Egreedy, **kwargs: expl_class(
    num_actions=env.action_space.n, **kwargs
)


# -------------------------------------------------------------
# replay buffers
# NOTE env: types.DiscreteEnv

# the dict arguements merging order overrides defaults when arguments
# are passed
make_online_memory = lambda env, **kwargs: memory.OutOfGraphReplayBuffer(
    **{**constants.default_memory_args, **constants.env_info(env), **kwargs}
)


# steps and iterations have defaults to pass replay_capacity in kwargs
# too - that is the only case when they are not required
make_offline_memory = lambda env, buffers_dir, buffers_iterations=None, parallel=False, **kwargs: memory.load_offline_buffers(
    buffers_dir=buffers_dir,
    iterations=buffers_iterations,
    parallel=parallel,
    **{
        **constants.default_memory_args,
        **constants.env_info(env),
        **kwargs,
    },
)


# -------------------------------------------------------------
# rng

make_rng = lambda seed: custom_pytrees.PRNGKeyWrap(seed)


# -------------------------------------------------------------
# reporters
make_reporters = lambda *reporter_confs: [
    reporter.AVAILABLE_REPORTERS[k](**v) for k, v in reporter_confs
]
