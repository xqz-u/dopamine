import logging
import os
import pprint

from thesis import constants, runner, utils

logger = logging.getLogger(__name__)


DEFAULT_REDUNDANCY = 3

DEFAULT_SEED = 42

dqn_cartpole_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)

dqn_acrobot_replay_buffers_root = os.path.join(
    str(constants.data_dir),
    "Acrobot-v1/DQNAgent/ab_dqn_full_experience_%%/checkpoints/full_experience",
)

for var_name, var in [
    ("dqn_cartpole_replay_buffers_root", dqn_cartpole_replay_buffers_root),
    ("dqn_acrobot_replay_buffers_root", dqn_acrobot_replay_buffers_root),
]:
    if not os.path.exists(var):
        logger.warning(f"Expected `{var_name}` at {var} does not exist!")


def run_experiment(run: runner.Runner):
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
