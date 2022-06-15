from typing import Any, Callable, Dict, Union

import gin
import gym
import optax
from dopamine.discrete_domains import atari_lib

# to be able to import stuff from here
from dopamine.discrete_domains.gym_lib import create_gym_environment
from dopamine.jax import losses

from thesis import agent, constants, exploration, memory, runner, types
from thesis.agent import utils as agent_utils

# This module contains functions which are supposed to avoid defining a
# myriad of constans in gin files, by instead configuring some functions
# which achieve the same job given a root parameter that can derive all
# the other required ones - these functions mostly depend on the chosen
# gym.Env.
# Moreover, this is the place to make callables gin-configurable if they
# are not already.


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


# NOTE optimizer_fn should have bound arguments
@gin.configurable
def create_model_TS_def(
    model_def: types.ModelDef,
    optimizer: optax.GradientTransformation,
    loss_fn: types.LossMetric,
) -> types.ModelTSDef:
    return (agent_utils.build_models(model_def), optimizer, loss_fn)


# NOTE not using **kwargs but `memory_args` since the latter can be
# bound with gin.
# if no `env` is given, the passed `memory_args` are merged with
# `constants.default_memory_args`, and the args specified in the latter
# are overwritten by the former.
# when `env` is passed, its dependent parameters are retrieved
# programmatically - this avoids many constant bindings in gin files
@gin.configurable
def create_memory(
    memory_call: memory.OutOfGraphReplayBuffer,
    memory_args: Dict[str, Any] = {},
    env: types.DiscreteEnv = None,
) -> memory.OutOfGraphReplayBuffer:
    memory_args = {**constants.default_memory_args, **memory_args}
    if env is None:
        return memory_call(**memory_args)
    return memory_call(**memory_args, **constants.env_info(env))


@gin.configurable
def create_explorer(
    explorer_call: exploration.PolicyEvaluator,
    explorer_args: Dict[str, Any] = {},
    env: types.DiscreteEnv = None,
) -> exploration.PolicyEvaluator:
    if env is None:
        return explorer_call(**explorer_args)
    assert "num_actions" not in explorer_args, "Duplicate key for `num_actions`"
    return explorer_call(num_actions=env.action_space.n, **explorer_args)


# NOTE assumes the args to the create_* functions and agent_call
# called here are already bound. exception: environment_* parameters
# are also given here, and override gin-defined ones, to be able to
# query them
@gin.configurable
def create_runner(
    runner_call: Union[runner.OnlineRunner, runner.FixedBatchRunner],
    agent_call: agent.Agent,
    create_env_fn: Callable[[str, str, Any], types.DiscreteEnv],
    environment_name: str,
    environment_version: str,
    steps: int,
    iterations: int,
    replay_capacity: int = None,
) -> Union[runner.OnlineRunner, runner.FixedBatchRunner]:
    env = create_env_fn(environment_name, environment_version)
    return runner_call(
        agent=agent_call(
            policy_evaluator=create_explorer(env=env),
            memory=create_memory(
                env=env,
                memory_args={
                    "replay_capacity": replay_capacity
                    if replay_capacity is not None
                    else steps * iterations
                },
            ),
        ),
        env=env,
        steps=steps,
        iterations=iterations,
    )
