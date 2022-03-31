import math
from typing import Union

import numpy as np
from dopamine.discrete_domains import atari_lib, gym_lib

opposite = lambda vs: tuple(map(lambda el: -el, vs))


def env_preproc_info(environment_name: str = None, version: str = None, **_) -> dict:
    return env_additional_info.get(f"{environment_name}-{version}", {})


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
