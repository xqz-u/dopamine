import math

opposite = lambda vs: tuple(map(lambda el: -el, vs))


def env_info(environment_name: str = None, version: str = None) -> dict:
    return env_configs.get(f"{environment_name}-{version}")


CARTPOLE_MIN_VALS = (-2.4, -5.0, -math.pi / 12.0, -math.pi * 2.0)

ACROBOT_MIN_VALS = (-1.0, -1.0, -1.0, -1.0, -5.0, -5.0)

MOUNTAINCAR_MIN_VALS = (-1.2, -0.07)
MOUNTAINCAR_MAX_VALS = (0.6, 0.07)

env_configs = {
    "CartPole-v1": {
        "observation_shape": (4, 1),
        "preproc": {
            "min_vals": CARTPOLE_MIN_VALS,
            "max_vals": opposite(CARTPOLE_MIN_VALS),
        },
    },
    "Acrobot-v1": {
        "observation_shape": (6, 1),
        "preproc": {
            "min_vals": ACROBOT_MIN_VALS,
            "max_vals": opposite(ACROBOT_MIN_VALS),
        },
    },
    "MountainCar-v0": {
        "observation_shape": (2, 1),
        "preproc": {
            "min_vals": MOUNTAINCAR_MIN_VALS,
            "max_vals": MOUNTAINCAR_MAX_VALS,
        },
    },
}
