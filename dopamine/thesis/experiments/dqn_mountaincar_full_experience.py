# generate training trajectories on the MountainCar Gym environment with
# a DQN agent
import pprint

from thesis import utils

utils.setup_root_logging()

from thesis import agent, configs, constants, runner

REDUNDANCY = 3
START_SEED = 501


def run_once(conf_args, conf_kwargs):
    conf_args[3] = {"Q_model_def": configs.adam_mse_mlp(2, "MountainCar-v0")}
    run = runner.OnlineRunner(
        **configs.make_online_runner_conf(*conf_args, **conf_kwargs)
    )
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


args = [
    (
        [
            "test_dqn_mountaincar_record_experience",
            # "dqn_mountaincar_record_experience",
            "MountainCar-v0",
            agent.DQN,
            # {"Q_model_def": configs.adam_mse_mlp(2, "MountainCar-v0")},
            START_SEED + i,
            i,
            # str(constants.scratch_data_dir),
            str(constants.data_dir),
        ],
        {"record_experience": True, "eval_period": 50, "steps": 500},
    )
    for i in range(REDUNDANCY)
]

import multiprocessing as mp

with mp.Pool() as pool:
    pool.starmap(run_once, args)
