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
from thesis.experiments import dqv_cartpole_acrobot_offline_gin

utils.setup_root_logging()


# NOTE loads all replay buffers iterations
def make_runner(
    experiment_name: str,
    env_name: str,
    seed: int,
    offline_data_dir: str,
    redundancy: int,
    logs_base_dir: str,
    iterations: int = 500,
    steps: int = 1000,
) -> runner.Runner:
    env = instantiators.create_gym_environment(*env_name.split("-"))
    logs_dir = utils.data_dir_from_conf(
        experiment_name, env_name, agent.DQV.__name__, logs_base_dir
    )
    make_model_def = lambda features: instantiators.create_model_TS_def(
        **{
            "model_def": (
                networks.MLP,
                {"features": features, **constants.env_preproc_info[env_name]},
            ),
            "optimizer": instantiators.adam(
                **{
                    "learning_rate": 0.001,
                    "eps": 3.125e-4,
                }
            ),
            "loss_fn": instantiators.mse_loss,
        }
    )
    # run = runner.FixedBatchRunner(
    return {
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
                    }
                ),
                "Q_model_def": make_model_def(2),
                "V_model_def": make_model_def(1),
            }
        ),
    }
    # )
    # run.run()


args = []
for i in range(dqv_cartpole_acrobot_offline_gin.REDUNDANCY):
    args.append(
        [
            "test_dqv_cartpole_offline",
            "CartPole-v1",
            dqv_cartpole_acrobot_offline_gin.CARTPOLE_START_SEED + i,
            dqv_cartpole_acrobot_offline_gin.dqn_cartpole_replay_buffers_root,
            i,
            # constants.data_dir,
            constants.scratch_data_dir,
        ]
    )
    args.append(
        [
            "test_dqv_acrobot_offline",
            "Acrobot-v1",
            dqv_cartpole_acrobot_offline_gin.ACROBOT_START_SEED + i,
            dqv_cartpole_acrobot_offline_gin.dqn_acrobot_replay_buffers_root,
            i,
            # constants.data_dir,
            constants.scratch_data_dir,
        ]
    )

if False:
    import multiprocessing as mp

    with mp.Pool(processes=len(args)) as pool:
        pool.starmap(make_runner, args)


aconf = make_runner(*args[0])
run = runner.FixedBatchRunner(**aconf)


def callable_name_getter(call_: callable):
    return getattr(call_, "__name__", type(call_).__name__)


def config_collector(obj: object, conf_dict: dict) -> dict:
    if not hasattr(obj, "reportable"):
        return conf_dict
    for field in obj.reportable:
        value = getattr(obj, field)
        if hasattr(value, "reportable"):
            conf_dict[field] = {
                "call_": callable_name_getter(value),
                **config_collector(value, {}),
            }
        else:
            conf_dict[field] = value
    return conf_dict


c = {}
c["agent"] = {
    "call_": callable_name_getter(run.agent),
    **config_collector(run.agent, {}),
}
