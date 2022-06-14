import itertools as it
import os
from typing import List

import gin
from thesis import constants, instantiators, runner, utils

utils.setup_root_logging()


# NOTE assumes a broader gin config was already parsed
def full_offline_runner(
    redundancy: int,
    seed: int,
    offline_data_dir: str,
    iterations: List[int],
    logs_dir: str,
    aim_dir: str,
):
    # bind an AimReporter too simply to avoid raising an error if it
    # won't be used anyways
    additional_bindings = f"""
    import thesis.reporter

    PRNGKeyWrap.seed = {seed}
    FixedBatchRunner.redundancy = {redundancy}
    FixedBatchRunner.checkpoint_base_dir = '{logs_dir}'
    load_offline_buffers.buffers_dir = '{offline_data_dir}'
    load_offline_buffers.iterations = {iterations}
    AimReporter.repo = '{aim_dir}'
    """
    with gin.unlock_config():
        gin.parse_config(additional_bindings)
    return instantiators.create_runner()


def make_offline_runners(
    gin_config_file: str,
    n: int,
    buffers_root_dir: str,
    first_seed: int = 42,
    intermediate_dirs: str = "",
    iterations: List[List[int]] = None,
    logs_base_dir: str = constants.data_dir,
) -> List[runner.Runner]:
    gin.parse_config_file(gin_config_file)
    logs_dir = utils.data_dir_from_conf(
        gin.query_parameter("%EXP_NAME"),
        "-".join(
            (
                gin.query_parameter("create_runner.environment_name"),
                gin.query_parameter("create_runner.environment_version"),
            )
        ),
        str(gin.query_parameter("create_runner.agent_call")),
        logs_base_dir,
    )
    return [
        full_offline_runner(redund, seed, buff_dir, iters_, logs_dir, logs_base_dir)
        for redund, seed, buff_dir, iters_ in zip(
            range(n),
            range(first_seed, first_seed + n),
            it.cycle(
                utils.unfold_replay_buffers_dir(buffers_root_dir, intermediate_dirs)
            ),
            it.cycle(iterations or ([None] * n)),
        )
    ]


dqn_data_dir = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)
gin_offline_config_file = os.path.join(
    str(constants.gin_configs_dir), "dqv_ens_cartpole_offline_test.gin"
)

runners = make_offline_runners(
    gin_offline_config_file,
    # 3,
    1,
    dqn_data_dir,
    iterations=[[1, 2, 3]],
    logs_base_dir=str(constants.scratch_data_dir),
)

# TODO take experiment_name out of config file
