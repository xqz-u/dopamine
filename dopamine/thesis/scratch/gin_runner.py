import logging
import os

import gin
from thesis import constants, instantiators, runner, utils
from thesis.agent import utils as agent_utils

log_level = logging.DEBUG
# log_level = logging.INFO
utils.setup_root_logging(log_level)
gin.enter_interactive_mode()


# NOTE since we want to have multiple redundancies, a runner must be
# created many times; each time, what changes is:
# * redundancy
# * [for offline_memory: the redundancy folder used for offline
#    trajectories]
# so, for multiple runs, do not configure these inside of gin
# +++++++
# if created inside gin:
#  - env should (better) be a singleton


gin_config_file = os.path.join(str(constants.scratch_dir), "gin_testfile.gin")
gin_offline_config_file = os.path.join(
    str(constants.scratch_dir), "gin_testfile_offline.gin"
)
dqn_data_dir = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
)


def make_run():
    gin.parse_config_file(gin_config_file)
    return instantiators.create_runner()


def runner_incomplete_gin_config():
    env_singleton_binding = """
        ENV = @shared_env/gin.singleton()
        shared_env/gin.singleton.constructor = @create_gym_environment

        create_explorer.env = %ENV
        create_memory.env = %ENV
        create_memory.memory_args = {"replay_capacity": 3600}

        DQVEnsemble.policy_evaluator = @create_explorer()
        DQVEnsemble.memory = @create_memory()

        OnlineRunner.agent = @DQVEnsemble()
        OnlineRunner.env = %ENV
        OnlineRunner.iterations = 12
        OnlineRunner.steps = 300
        OnlineRunner.schedule = "train"
    """
    gin.parse_config_files_and_bindings(
        config_files=[gin_config_file], bindings=[env_singleton_binding]
    )
    return runner.OnlineRunner()


def offline_runner():
    gin.parse_config_file(gin_offline_config_file)
    with gin.unlock_config():
        gin.bind_parameter("load_offline_buffers.buffers_dir", dqn_data_dir)
    return instantiators.create_runner()


# run = make_run()
# run.run()

run = offline_runner()
# run.run()
