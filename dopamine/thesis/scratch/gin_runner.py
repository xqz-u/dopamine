import os

import gin
from thesis import agent, constants, instantiators, memory, runner, utils
from thesis.agent import utils as agent_utils
from thesis.memory import offline_memory

utils.setup_root_logging()
gin.enter_interactive_mode()

dqn_data_dir = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
)

env = instantiators.create_gym_environment("CartPole", "v1")


off_mem = memory.OutOfGraphReplayBuffer(
    **constants.env_info(env),
    **constants.default_memory_args,
)

off_mem.load(dqn_data_dir, 9)

off_mem._store["observation"].shape


# -----------------------------


# TODO change all references to OfflineOutOfGraphReplayBuffer to
#      load_offline_buffers
# TODO load the new function directly in memory module
offline_mem = offline_memory.load_offline_buffers(
    buffers_dir=dqn_data_dir,
    iterations=[0, 1, 2],
    **constants.env_info(env),
    **constants.default_memory_args,
)


agent_utils.sample_replay_buffer(offline_mem)

offline_mem._store["observation"]
offline_mem._store["action"]

m = instantiators.create_memory(
    memory_class=memory.OfflineOutOfGraphReplayBuffer,
    env=env,
    memory_args={
        "_buffers_dir": f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0",
        "_buffers_iterations": [0],
        # "load_parallel": False,
    },
)

m._store["observation"]
m._store["action"]


# -----------------------------


# TODO test offline runner
# TODO preproc values to networks
# TODO config collection from gin

# NOTE since we want to have multiple redundancies, a runner must be
# created many times; each time, what changes is:
# * redundancy
# * [for offline_memory: the redundancy folder used for offline
#    trajectories]
# so, for multiple runs, do not configure these inside of gin
# +++++++
# if created inside gin:
#  - env should (better) be a singleton


ENV_SINGLETON_BINDING = """
    ENV = @shared_env/gin.singleton()
    shared_env/gin.singleton.constructor = @create_gym_environment

    create_explorer.env = %ENV
    create_memory.env = %ENV

    DQVEnsemble.policy_evaluator = @create_explorer()
    DQVEnsemble.memory = @create_memory()

    OnlineRunner.agent = @DQVEnsemble()
    OnlineRunner.env = %ENV
"""

gin_config_file = os.path.join(str(constants.scratch_dir), "gin_testfile.gin")
gin_offline_config_file = os.path.join(
    str(constants.scratch_dir), "gin_testfile_offline.gin"
)


def runner_incomplete_gin_config():
    gin.parse_config_file(gin_config_file)
    env = instantiators.create_gym_environment()
    return runner.OnlineRunner(
        agent=agent.DQVEnsemble(
            policy_evaluator=instantiators.create_explorer(env=env),
            memory=instantiators.create_memory(env=env),
        ),
        env=env,
    )


def runner_complete_gin_config():
    gin.parse_config_files_and_bindings(
        config_files=[gin_config_file], bindings=[ENV_SINGLETON_BINDING]
    )
    return runner.OnlineRunner()


def offline_runner():
    gin.parse_config_file(gin_offline_config_file)
    env = instantiators.create_gym_environment()
    return runner.FixedBatchRunner(
        agent=agent.DQVEnsemble(
            policy_evaluator=instantiators.create_explorer(env=env),
            memory=instantiators.create_memory(
                env=env,
                memory_args={
                    "_buffers_dir": f"{constants.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience/0"
                },
            ),
        ),
        env=env,
    )


run = offline_runner()
