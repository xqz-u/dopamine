import os

import gin
from thesis import agent, constants, instantiators, memory, runner, utils

utils.setup_root_logging()

gin.enter_interactive_mode()

gin_config_file = os.path.join(str(constants.scratch_dir), "gin_testfile.gin")
gin.parse_config_file(gin_config_file)
with gin.unlock_config():
    gin.bind_parameter("OnlineRunner.reporters", [])

env = instantiators.create_gym_environment()

run = runner.OnlineRunner(
    agent=agent.DQVEnsemble(
        policy_evaluator=instantiators.create_explorer(env=env),
        memory=instantiators.create_memory(env=env),
    ),
    env=env,
)
# run.run()
