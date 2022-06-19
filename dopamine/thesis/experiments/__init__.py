import logging
import os
from typing import Callable, Dict

import gin
import gym
from thesis import agent, configs, constants, types, utils
from thesis.agent import utils as agent_utils

logger = logging.getLogger(__name__)


DEFAULT_REDUNDANCY = 3

DEFAULT_SEED = 42

dqn_cartpole_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)

dqn_acrobot_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "Acrobot-v1/DQNAgent/ab_dqn_full_experience_%%/checkpoints/full_experience",
)

for var_name, var in [
    ("dqn_cartpole_replay_buffers_root", dqn_cartpole_replay_buffers_root),
    ("dqn_acrobot_replay_buffers_root", dqn_acrobot_replay_buffers_root),
]:
    if not os.path.exists(var):
        logger.warning(f"Expected `{var_name}` at {var} does not exist!")


# NOTE runner_args are:
# record_experience: bool
# iterations: int
# steps: int
# eval_steps: int
# eval_period: int
@gin.configurable
def make_conf(
    experiment_name: str,
    env_name: str,
    agent_class: agent.Agent,
    model_maker_fn: Callable[[types.DiscreteEnv], Dict[str, agent_utils.ModelDefStore]],
    logs_base_dir: str,
    seed: int,
    redundancy: int,
    env_creator_fn: Callable[[str, str], types.DiscreteEnv] = gym.make,
    offline_root_data_dir: str = None,
    **kwargs,
) -> dict:
    env = env_creator_fn(env_name)
    logs_dir = utils.data_dir_from_conf(
        experiment_name,
        env_name,
        utils.callable_name_getter(agent_class),
        logs_base_dir,
    )
    return {
        **kwargs["experiment"],
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": configs.make_reporters(
            (
                "mongo",
                {
                    "experiment_name": experiment_name,
                    **kwargs.get("reporters", {}).get("mongo", {}),
                },
            ),
            (
                "aim",
                {
                    "experiment_name": experiment_name,
                    "repo": str(logs_base_dir),
                    **kwargs.get("reporters", {}).get("aim", {}),
                },
            ),
        ),
        "on_policy_eval": [agent_utils.t0_max_q_callback],
        "agent": agent_class(
            **{
                **kwargs.get("agent", {}),
                "rng": configs.make_rng(seed),
                "policy_evaluator": configs.make_explorer(env),
                "memory": (
                    configs.make_offline_memory(
                        env,
                        os.path.join(offline_root_data_dir, str(redundancy)),
                        replay_capacity=kwargs["experiment"]["steps"]
                        * kwargs["experiment"]["iterations"],
                        **kwargs.get("memory", {}),
                    )
                    if offline_root_data_dir
                    else configs.make_online_memory(env)
                ),
                **model_maker_fn(env_name, env.action_space.n),
            }
        ),
    }
