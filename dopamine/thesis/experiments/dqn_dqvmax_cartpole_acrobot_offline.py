import os
from typing import Callable, Dict

import gin
from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner, types
from thesis.agent import utils as agent_utils


# to turn lambda list in keywords
@gin.configurable
def make_conf(
    experiment_name: str,
    env_name: str,
    agent_class: agent.Agent,
    model_maker_fn: Callable[[types.DiscreteEnv], Dict[str, agent_utils.ModelDefStore]],
    offline_root_data_dir: str,
    logs_base_dir: str,
    seed: int,
    redundancy: int,
) -> dict:
    env = configs.create_gym_environment(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name,
        env_name,
        utils.callable_name_getter(agent_class),
        logs_base_dir,
    )
    steps, iterations, eval_period = 500, 1000, 10
    return {
        "iterations": iterations,
        "steps": steps,
        "eval_steps": steps,
        "eval_period": eval_period,
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": configs.make_reporters(
            ("mongo", {"experiment_name": experiment_name}),
            ("aim", {"experiment_name": experiment_name, "repo": str(logs_base_dir)}),
        ),
        "on_policy_eval": [agent_utils.t0_max_q_callback],
        "agent": agent_class(
            **{
                "rng": configs.make_rng(seed),
                "policy_evaluator": configs.make_explorer(env),
                "memory": configs.make_offline_memory(
                    env,
                    os.path.join(offline_root_data_dir, str(redundancy)),
                    replay_capacity=steps * iterations,
                ),
                **model_maker_fn(env_name, env.environment.action_space.n),
            }
        ),
    }


def make_conf_and_run(conf_args: dict):
    experiments.run_experiment(runner.FixedBatchRunner(**make_conf(**conf_args)))


confs = [
    c
    for agent_class, model_maker in [
        [agent.DQN, configs.dqn_model_maker],
        [agent.DQVMax, configs.dqvmax_model_maker],
    ]
    for env_name, offline_buff_dir in [
        ["CartPole-v1", experiments.dqn_cartpole_replay_buffers_root],
        ["Acrobot-v1", experiments.dqn_acrobot_replay_buffers_root],
    ]
    for c in [
        {
            "seed": experiments.DEFAULT_SEED + i,
            "redundancy": i,
            "agent_class": agent_class,
            "env_name": env_name,
            "offline_root_data_dir": offline_buff_dir,
            "experiment_name": f"{utils.callable_name_getter(agent_class)}_{env_name}_offline",
            "model_maker_fn": model_maker,
            "logs_base_dir": constants.data_dir,
        }
        for i in range(experiments.DEFAULT_REDUNDANCY)
    ]
]


if __name__ == "__main__":
    import multiprocessing as mp

    with mp.Pool() as pool:
        res = pool.map(make_conf_and_run, confs)
