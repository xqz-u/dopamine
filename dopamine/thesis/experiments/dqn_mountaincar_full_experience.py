# generate training trajectories on the MountainCar Gym environment with
# a DQN agent
from typing import Dict

from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, experiments, runner
from thesis.agent import utils as agent_utils

START_SEED = 501


# TODO
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
    env_creator: callable = configs.create_gym_environment,
    **runner_kwargs,
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
        "redundancy": redundancy,
        "experiment_name": experiment_name,
        "env": env,
        "checkpoint_base_dir": logs_dir,
        "reporters": configs.make_reporters(
            ("mongo", {"experiment_name": experiment_name}),
            ("aim", {"experiment_name": experiment_name, "repo": str(logs_base_dir)}),
        ),
        "agent": agent_class(
            **{
                "rng": configs.make_rng(seed),
                "policy_evaluator": configs.make_explorer(env),
                "memory": configs.make_online_memory(env),
                **models_dict,
            }
        ),
    }


def run_once(conf_args, conf_kwargs):
    # NOTE put here MLP.activation_fn is not pickle-serializable -
    # necessary for pool.map
    conf_args.insert(3, {"Q_model_def": configs.adam_mse_mlp(2, "MountainCar-v0")})
    experiments.run_experiment(
        runner.OnlineRunner(**make_online_runner_conf(*conf_args, **conf_kwargs))
    )


args = [
    (
        [
            # "test_dqn_mountaincar_record_experience",
            "dqn_mountaincar_record_experience",
            "MountainCar-v0",
            agent.DQN,
            # {"Q_model_def": configs.adam_mse_mlp(2, "MountainCar-v0")},
            START_SEED + i,
            i,
            str(constants.scratch_data_dir),
            # str(constants.data_dir),
        ],
        {"record_experience": True, "eval_period": 50, "steps": 500},
    )
    for i in range(experiments.DEFAULT_REDUNDANCY)
]


if __name__ == "__main__":
    import multiprocessing as mp

    with mp.Pool() as pool:
        pool.starmap(run_once, args)
