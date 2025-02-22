import math
import os
from pathlib import Path
from typing import Union

import numpy as np
from dopamine.discrete_domains import atari_lib, gym_lib

base_dir = Path(os.path.dirname(__file__))
dopamine_dir = Path(base_dir.parent, "dopamine")

data_dir = Path(base_dir.parent.parent, "resources", "data")
aim_dir = data_dir
scratch_data_dir = data_dir.joinpath("scratch")
peregrine_data_dir = "/data/s3680622"

scratch_dir = base_dir.joinpath("scratch")

default_memory_args = {"replay_capacity": int(1e6), "batch_size": 32}

opposite = lambda vs: tuple(map(lambda el: -el, vs))


def env_preproc_info(environment_name: str = None, version: str = None, **_) -> dict:
    return env_additional_info.get(f"{environment_name}-{version}", {})


# NOTE when an agent is created, the relevant environment information
# are passed from this function, regardless whether they were
# specified in the experiment config; such info are mostly important
# for the replay buffers. #actions is not passed to keep
# initialization compatible with both Agent and Memory (the latter
# would be invalidated)
def env_info(
    env: Union[gym_lib.GymPreprocessing, atari_lib.AtariPreprocessing]
) -> dict:
    return dict(
        zip(
            ["observation_shape", "observation_dtype", "stack_size"],
            (
                [
                    atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                    # atari_lib.NATURE_DQN_DTYPE,
                    np.uint8,
                    atari_lib.NATURE_DQN_STACK_SIZE,
                ]
                if isinstance(env, atari_lib.AtariPreprocessing)
                else [
                    env.observation_space.shape + (1,),
                    env.observation_space.dtype,
                    1,
                ]
            ),
        )
    )


CARTPOLE_MIN_VALS = (-2.4, -5.0, -math.pi / 12.0, -math.pi * 2.0)

ACROBOT_MIN_VALS = (-1.0, -1.0, -1.0, -1.0, -5.0, -5.0)

MOUNTAINCAR_MIN_VALS = (-1.2, -0.07)
MOUNTAINCAR_MAX_VALS = (0.6, 0.07)

env_additional_info = {
    "CartPole-v1": {
        "preproc": {
            "min_vals": CARTPOLE_MIN_VALS,
            "max_vals": opposite(CARTPOLE_MIN_VALS),
        },
    },
    "Acrobot-v1": {
        "preproc": {
            "min_vals": ACROBOT_MIN_VALS,
            "max_vals": opposite(ACROBOT_MIN_VALS),
        },
    },
    "MountainCar-v0": {
        "preproc": {
            "min_vals": MOUNTAINCAR_MIN_VALS,
            "max_vals": MOUNTAINCAR_MAX_VALS,
        },
    },
}
