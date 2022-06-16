import os
import pprint

import optax
from dopamine.jax import losses
from thesis import (
    agent,
    configs,
    constants,
    custom_pytrees,
    experiments,
    exploration,
    memory,
    networks,
    reporter,
    runner,
    utils,
)
from thesis.agent import utils as agent_utils

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
    env = configs.create_gym_environment(*env_name.split("-"))
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
            "opt": optax.adam,
            "opt_params": {
                "learning_rate": 0.001,
                "eps": 3.125e-4,
            },
            "loss_fn": losses.mse_loss,
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
            reporter.MongoReporter(experiment_name=experiment_name),
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


CARTPOLE_START_SEED = 12
ACROBOT_START_SEED = CARTPOLE_START_SEED
REDUNDANCY = 3


args = []
for i in range(REDUNDANCY):
    args.append(
        [
            # "test_dqv_cartpole_offline",
            "dqv_cartpole_offline",
            "CartPole-v1",
            CARTPOLE_START_SEED + i,
            experiments.dqn_cartpole_replay_buffers_root,
            i,
            constants.data_dir,
            # constants.scratch_data_dir,
        ]
    )
    args.append(
        [
            # "test_dqv_acrobot_offline",
            "dqv_acrobot_offline",
            "Acrobot-v1",
            ACROBOT_START_SEED + i,
            experiments.dqn_acrobot_replay_buffers_root,
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


if __name__ == "__main__":
    import multiprocessing as mp

    with mp.Pool() as pool:
        pool.map(run_once, [arg for arg in args])
