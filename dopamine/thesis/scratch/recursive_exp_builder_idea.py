import os

from thesis import (
    agent,
    constants,
    custom_pytrees,
    exploration,
    instantiators,
    memory,
    networks,
    reporter,
    runner,
    utils,
)
from thesis.agent import utils as agent_utils


def make_conf(
    experiment_name: str,
    env_name: str,
    seed: int,
    offline_data_dir: str,
    redundancy: int,
    iterations: int = 500,
    steps: int = 1000,
    logs_base_dir: str = constants.data_dir,
):
    env = instantiators.create_gym_environment(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name, env_name, agent.DQV.__name__, logs_base_dir
    )
    return {
        "runner": {
            "call_": runner.FixedBatchRunner,
            "iterations": iterations,
            "steps": steps,
            "eval_steps": steps,
            "eval_period": 5,
            "redundancy": redundancy,
            "env": env,
            "checkpoint_base_dir": logs_dir,
            "reporters": [
                reporter.MongoReporter(experiment_name=experiment_name),
                reporter.AimReporter(
                    experiment_name=experiment_name,
                    repo=str(logs_base_dir),
                ),
            ],
            "on_policy_eval": [agent_utils.t0_max_q_callback],
            "agent": {
                "call_": agent.DQV,
                "rng": {"call_": custom_pytrees.PRNGKeyWrap, "seed": seed},
                "policy_evaluator": {
                    "call_": exploration.Egreedy,
                    "num_actions": env.environment.action_space.n,
                },
                "memory": {
                    "call_": memory.load_offline_buffers,
                    **constants.env_info(env),
                    "replay_capacity": steps * iterations,
                    "buffers_dir": offline_data_dir,
                },
                "Q_model_def": {
                    "call_": instantiators.create_model_TS_def,
                    "model_def": (
                        networks.MLP,
                        {"features": 2, **constants.env_preproc_info[env_name]},
                    ),
                    "optimizer": {
                        "call_": instantiators.adam,
                        "learning_rate": 0.001,
                        "eps": 3.125e-4,
                    },
                    "loss_fn": instantiators.mse_loss,
                },
                "V_model_def": {
                    "call_": instantiators.create_model_TS_def,
                    "model_def": (
                        networks.MLP,
                        {"features": 1, **constants.env_preproc_info[env_name]},
                    ),
                    "optimizer": {
                        "call_": instantiators.adam,
                        "learning_rate": 0.001,
                        "eps": 3.125e-4,
                    },
                    "loss_fn": instantiators.mse_loss,
                },
            },
        }
    }


dqn_cartpole_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)


conf = make_conf(
    "pippo",
    "CartPole-v1",
    7,
    dqn_cartpole_replay_buffers_root,
    0,
    logs_base_dir=constants.scratch_data_dir,
)


# def inner(a=1, b=2):
#     return a + b


# def outer(d, c=3):
#     print(f"received: {c} {d}")
#     return c + d


# test = {"call_": outer, "c": {"call_": inner, "a": 5, "b": 6}, "d": 2}
