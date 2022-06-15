# train a dqv agent offline on the CartPole and AcroBot environments
# using replay trajectories produced by a DQN agent during training

import os

import gin
from thesis import constants, networks, runner, utils
from thesis.runner import utils as runner_utils

utils.setup_root_logging()

dqv_cartpole_offline_conf = os.path.join(
    str(constants.gin_configs_dir), "dqv_offline.gin"
)
dqn_cartpole_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)
dqn_acrobot_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "Acrobot-v1/DQNAgent/ab_dqn_full_experience_%%/checkpoints/full_experience",
)


def bind_env_related_params(env, version):
    environment = f"{env}-{version}"
    classic_control_mlp = lambda features: (
        networks.MLP,
        {"features": features, **constants.env_preproc_info[environment]},
    )
    with gin.unlock_config():
        gin.bind_parameter(
            "vfunc/create_model_TS_def.model_def", classic_control_mlp(1)
        )
        gin.bind_parameter(
            "qfunc/create_model_TS_def.model_def", classic_control_mlp(2)
        )
        gin.bind_parameter("create_runner.environment_name", env)
        gin.bind_parameter("create_runner.environment_version", version)


CARTPOLE_START_SEED = 12
ACROBOT_START_SEED = CARTPOLE_START_SEED
REDUNDANCY = 3

if False:

    gin.parse_config_file(dqv_cartpole_offline_conf)

    bind_env_related_params("CartPole", "v1")
    runs = runner_utils.make_offline_runners(
        "test_dqv_cartpole_offline",
        1,
        # REDUNDANCY,
        dqn_cartpole_replay_buffers_root,
        CARTPOLE_START_SEED,
        logs_base_dir=constants.scratch_data_dir,
    )

    bind_env_related_params("Acrobot", "v1")
    runs += runner_utils.make_offline_runners(
        "test_dqv_acrobot_offline",
        REDUNDANCY,
        dqn_acrobot_replay_buffers_root,
        ACROBOT_START_SEED,
        logs_base_dir=constants.scratch_data_dir,
    )


# import multiprocessing as mp


# def run_once(run: runner.Runner):
#     run.run()


# with mp.Pool(processes=len(runs)) as pool:
#     pool.map(run_once, runs)
