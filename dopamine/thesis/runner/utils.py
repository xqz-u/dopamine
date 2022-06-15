import itertools as it
from typing import List

import gin
from thesis import constants, instantiators, runner, utils


# NOTE assumes a broader gin config was already parsed
def full_offline_runner(
    experiment_name: str,
    redundancy: int,
    seed: int,
    offline_data_dir: str,
    iterations: List[int],
    logs_dir: str,
    aim_dir: str,
):
    # bind an AimReporter too simply to avoid raising an error if it
    # is not used anyways
    additional_bindings = f"""
    import thesis.reporter

    EXP_NAME = '{experiment_name}'

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
    experiment_name: str,
    n: int,
    buffers_root_dir: str,
    first_seed: int = 42,
    intermediate_dirs: str = "",
    iterations: List[List[int]] = None,
    logs_base_dir: str = constants.data_dir,
) -> List[runner.Runner]:
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
        full_offline_runner(
            experiment_name, redund, seed, buff_dir, iters_, logs_dir, logs_base_dir
        )
        for redund, seed, buff_dir, iters_ in zip(
            range(n),
            range(first_seed, first_seed + n),
            it.cycle(
                utils.unfold_replay_buffers_dir(buffers_root_dir, intermediate_dirs)
            ),
            it.cycle(iterations or ([None] * n)),
        )
    ]
