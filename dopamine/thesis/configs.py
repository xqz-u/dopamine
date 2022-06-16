import os
from typing import Dict

import optax
from dopamine.jax import losses

from thesis import (agent, constants, custom_pytrees, exploration,
                    instantiators, memory, networks, reporter, utils)
from thesis.agent import utils as agent_utils

# -------------------------------------------------------------
# networks, loss metrics, optimizers definitions
# (use as agent_utils.ModelDefStore)

make_mlp_def = (
    lambda features, env_name, **kwargs: (
        networks.MLP,
        {
            "features": features,
            **constants.env_preproc_info.get(env_name, {}),
            **kwargs,
        },
    )
)


make_ensemble_def = lambda n_heads, heads_model: (
    networks.EnsembledNet,
    {"n_heads": n_heads, "model": heads_model},
)


make_adam_mse_def = lambda: {
    "opt": instantiators.adam,
    # "opt": optax.adam,
    "opt_params": {
        "learning_rate": 0.001,
        "eps": 3.125e-4,
    },
    "loss_fn": instantiators.mse_loss,
    # "loss_fn": losses.mse_loss,
}


adam_mse_mlp = lambda features, env_name: agent_utils.ModelDefStore(
    **{"net_def": make_mlp_def(features, env_name), **make_adam_mse_def()}
)


adam_mse_ensemble_mlp = lambda n_heads, features, env_name: agent_utils.ModelDefStore(
    **{
        "net_def": make_ensemble_def(n_heads, make_mlp_def(features, env_name)),
        **make_adam_mse_def(),
    }
)


# -------------------------------------------------------------
# policy evaluators, default to Egreedy
make_explorer = (
    lambda num_actions, expl_class=exploration.Egreedy, **kwargs: expl_class(
        num_actions=num_actions, **kwargs
    )
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
make_offline_memory = lambda env, buffers_dir, buffers_iterations=None, steps=0, iterations=0, parallel=False, **kwargs: memory.load_offline_buffers(
    buffers_dir=buffers_dir,
    iterations=buffers_iterations,
    parallel=parallel
    ** {
        **constants.default_memory_args,
        **constants.env_info(env),
        **{"replay_capacity": steps * iterations},
        **kwargs,
    },
)


# -------------------------------------------------------------
# rng

make_rng = lambda seed: custom_pytrees.PRNGKeyWrap(seed)


# -------------------------------------------------------------
# reporters, makes all currently available ones by default

make_reporters = lambda exp_name, aim_repo_dir: [
    reporter.MongoReporter(experiment_name=exp_name),
    reporter.AimReporter(
        experiment_name=exp_name,
        repo=str(aim_repo_dir),
    ),
]


# -------------------------------------------------------------
# full configuration factories
# NOTE these are simple functions which make many parameters
# default, and accept only a limited number of configurable
# parameters; gin's or sacred's approach of partialling or composing
# configurations is better and more general. use these simple
# functions as long as only limited configurability is desired, can
# also break them in smaller functions, they are almost equal


# NOTE when schedule == 'train_and_eval' and full_experience, give high
# eval_period for better efficiency
def make_online_runner_conf(
    experiment_name: str,
    env_name: str,
    agent_class: agent.Agent,
    models_dict: Dict[str, agent_utils.ModelDefStore],
    seed: int,
    redundancy: int,
    logs_base_dir: str = constants.scratch_data_dir,
    iterations: int = 1000,
    steps: int = 600,
    env_creator: callable = instantiators.create_gym_environment,
    **runner_kwargs
) -> dict:
    env = env_creator(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name,
        env_name,
        utils.callable_name_getter(agent_class),
        logs_base_dir,
    )
    return {
        **runner_kwargs,
        "iterations": iterations,
        "steps": steps,
        "eval_steps": steps,
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": make_reporters(experiment_name, logs_base_dir),
        "agent": agent_class(
            **{
                "rng": make_rng(seed),
                "policy_evaluator": make_explorer(env.environment.action_space.n),
                "memory": make_online_memory(env),
                **models_dict,
            }
        ),
    }


# NOTE loads all replay buffers iterations
def make_offline_runner_conf(
    experiment_name: str,
    env_name: str,
    agent_class: agent.Agent,
    models_dict: Dict[str, agent_utils.ModelDefStore],
    seed: int,
    offline_data_dir: str,
    redundancy: int,
    logs_base_dir: str = constants.scratch_data_dir,
    iterations: int = 1000,
    steps: int = 600,
    env_creator: callable = instantiators.create_gym_environment,
    **runner_kwargs
) -> dict:
    env = env_creator(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name,
        env_name,
        utils.callable_name_getter(agent_class),
        logs_base_dir,
    )
    redundancy_offline_data_dir = os.path.join(offline_data_dir, str(redundancy))
    return {
        **runner_kwargs,
        "iterations": iterations,
        "steps": steps,
        "eval_steps": steps,
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": make_reporters(experiment_name, logs_base_dir),
        "on_policy_eval": [agent_utils.t0_max_q_callback],
        "agent": agent_class(
            **{
                "rng": make_rng(seed),
                "policy_evaluator": make_explorer(env.environment.action_space.n),
                "memory": make_offline_memory(
                    env,
                    redundancy_offline_data_dir,
                    steps=steps,
                    iterations=iterations,
                ),
                **models_dict,
            }
        ),
    }
