"""
This module contains functions which are supposed to avoid defining a
myriad of constans in gin files, by instead configuring some functions
which achieve the same job given a root parameter that can derive all
the other required ones.
Moreover, this is the place to make callables gin-configurable if they
are not already.
"""
from typing import Any, Dict, Union

import gin
import gym
import optax
from dopamine.discrete_domains import atari_lib

# to be able to import stuff from here
from dopamine.discrete_domains.gym_lib import create_gym_environment
from dopamine.jax import losses

from thesis import constants, exploration, memory, types
from thesis.agent import utils as agent_utils

adam = gin.configurable(optax.adam)
mse_loss = gin.configurable(losses.mse_loss)


# see dopamine.discrete_domains.atari_lib.create_atari_environment's
# docs for more comments; rewrote here to allow creation of environments
# under the ALE/ namespace too
@gin.configurable
def create_atari_environment(
    environment_name: str, version: str, env_args: Dict[str, Any] = None
):
    return atari_lib.AtariPreprocessing(gym.make(f"{environment_name}-{version}").env)


@gin.configurable
def create_model_TS_def(
    model_def: types.ModelDef,
    opt: optax.GradientTransformation,
    loss_fn: types.LossMetric,
) -> types.ModelTSDef:
    return (agent_utils.build_models(model_def), opt, loss_fn)


# NOTE not using **kwargs but `memory_args` since the latter can be
# bound with gin.
# if no `env` is given, the passed `memory_args` are merged with
# `constants.default_memory_args`, and the args specified in the latter
# are overwritten by the former.
# when `env` is passed, its dependent parameters are retrieved
# programmatically - this avoids many constant bindings in gin files
@gin.configurable
def create_memory(
    memory_class: Union[
        memory.OutOfGraphReplayBuffer, memory.OfflineOutOfGraphReplayBuffer
    ],
    memory_args: Dict[str, Any] = {},
    env: types.DiscreteEnv = None,
) -> Union[memory.OutOfGraphReplayBuffer, memory.OfflineOutOfGraphReplayBuffer]:
    memory_args = {**constants.default_memory_args, **memory_args}
    if env is None:
        return memory_class(**memory_args)
    return memory_class(**memory_args, **constants.env_info(env))


@gin.configurable
def create_explorer(
    explorer_class: exploration.PolicyEvaluator,
    explorer_args: Dict[str, Any] = {},
    env: types.DiscreteEnv = None,
) -> exploration.PolicyEvaluator:
    if env is None:
        return explorer_class(**explorer_args)
    assert "num_actions" not in explorer_args, "Duplicate key for `num_actions`"
    return explorer_class(num_actions=env.action_space.n, **explorer_args)
