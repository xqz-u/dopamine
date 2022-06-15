import os
import pprint

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
from thesis.scratch import dqv_cartpole_acrobot_offline_gin

utils.setup_root_logging()


# NOTE loads all replay buffers iterations
def make_runner_conf(
    experiment_name: str,
    env_name: str,
    seed: int,
    offline_data_dir: str,
    redundancy: int,
    logs_base_dir: str,
    iterations: int = 1000,
    steps: int = 600,
) -> runner.Runner:
    env = instantiators.create_gym_environment(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name, env_name, agent.DQV.__name__, logs_base_dir
    )
    make_model_def = lambda features, env_name: agent_utils.ModelDefStore(
        **{
            "net_def":
            # (
            #     networks.EnsembledNet,
            #     {
            #         "n_heads": 3,
            #         "model":
            (
                networks.MLP,
                {"features": features, **constants.env_preproc_info[env_name]},
            ),
            #     },
            # )
            "opt": instantiators.adam,
            "opt_params": {
                "learning_rate": 0.001,
                "eps": 3.125e-4,
            },
            "loss_fn": instantiators.mse_loss,
        }
    )
    return {
        "iterations": iterations,
        "steps": steps,
        "eval_steps": steps,
        "eval_period": 5,
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": [
            reporter.MongoReporter(
                experiment_name=experiment_name, metrics_buffering=4
            ),
            reporter.AimReporter(
                experiment_name=experiment_name,
                repo=str(logs_base_dir),
            ),
        ],
        "on_policy_eval": [agent_utils.t0_max_q_callback],
        "agent": agent.DQV(
            **{
                "rng": custom_pytrees.PRNGKeyWrap(seed),
                "policy_evaluator": exploration.Egreedy(
                    **{"num_actions": env.environment.action_space.n}
                ),
                "memory": memory.load_offline_buffers(
                    **constants.env_info(env),
                    **{
                        "replay_capacity": steps * iterations,
                        "batch_size": 32,
                        "buffers_dir": os.path.join(offline_data_dir, str(redundancy)),
                    },
                ),
                "Q_model_def": make_model_def(2, env_name),
                "V_model_def": make_model_def(1, env_name),
            }
        ),
    }


args = []
for i in range(dqv_cartpole_acrobot_offline_gin.REDUNDANCY):
    args.append(
        [
            "test_dqv_cartpole_offline",
            "CartPole-v1",
            dqv_cartpole_acrobot_offline_gin.CARTPOLE_START_SEED + i,
            dqv_cartpole_acrobot_offline_gin.dqn_cartpole_replay_buffers_root,
            i,
            constants.data_dir,
            # constants.scratch_data_dir,
        ]
    )
    args.append(
        [
            "test_dqv_acrobot_offline",
            "Acrobot-v1",
            dqv_cartpole_acrobot_offline_gin.ACROBOT_START_SEED + i,
            dqv_cartpole_acrobot_offline_gin.dqn_acrobot_replay_buffers_root,
            i,
            constants.data_dir,
            # constants.scratch_data_dir,
        ]
    )


def run_once(conf_args):
    run = runner.FixedBatchRunner(**make_runner_conf(*conf_args))
    c = utils.reportable_config(
        {
            "call_": utils.callable_name_getter(run),
            **utils.config_collector(run, "reportable"),
        }
    )
    pprint.pprint(c)
    for rep in run.reporters:
        rep.register_conf(c)
    run.run()


import multiprocessing as mp

with mp.Pool() as pool:
    pool.map(run_once, [arg for arg in args])
