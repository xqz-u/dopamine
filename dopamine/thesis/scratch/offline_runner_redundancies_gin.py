import os

import gin
from thesis import constants, utils
from thesis.runner import utils as runner_utils

utils.setup_root_logging()


dqn_data_dir = os.path.join(
    str(constants.data_dir),
    "CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/full_experience",
)
gin_offline_config_file = os.path.join(
    str(constants.gin_configs_dir), "dqv_ens_cartpole_offline_test.gin"
)

gin.parse_config_file(gin_offline_config_file)
with gin.unlock_config():
    gin.bind_parameter("MongoReporter.db_name", "thesis_test")

runners = runner_utils.make_offline_runners(
    "test_experiment",
    # 3,
    1,
    dqn_data_dir,
    iterations=[[1, 2, 3]],
    logs_base_dir=str(constants.scratch_data_dir),
)

run = runners[0]
# run.run()

import inspect
from typing import List

import attrs
from thesis import runner


def update_args_dict(instance: object, args_dict: dict):
    ret = args_dict.copy()
    for desired_field in instance.reportable_fields:
        if desired_field not in args_dict:
            ret[desired_field] = getattr(instance, desired_field)
    return ret


from jax import random as jrand


class Base:
    @classmethod
    def from_dict(cls, args_dict: dict):
        print(f"instantiate {cls} with: {args_dict}")
        instance = cls(**args_dict)
        new_dict = update_args_dict(instance, args_dict)
        print(f"updated args_dict: {new_dict}")
        return instance, new_dict


@attrs.define
class CustomRNG(Base):
    seed: int = 42
    key: jrand.KeyArray = None
    n_splits: int = 0

    def __post_init__(self):
        if self.key is None:
            self.reset()

    def __next__(self) -> jrand.KeyArray:
        self.key, sk = jrand.split(self.key)
        self.n_splits += 1
        return sk

    def reset(self):
        # logger.debug(f"Reset {self}")
        self.key = jrand.PRNGKey(self.seed)
        self.n_splits = 0
        # logger.debug(f"After reset: {self}")

    @property
    def reportable_fields(self):
        return ["seed"]


@attrs.define
class Egreedy(Base):
    num_actions: int
    epsilon_train: float = 0.01
    epsilon_eval: float = 0.001

    @property
    def reportable_fields(self):
        return [a.name for a in self.__attrs_attrs__]


@attrs.define
class DQV(Base):
    rng: CustomRNG
    policy_evaluator: Egreedy
    steps: int = 300

    @property
    def reportable_fields(self):
        return ["steps"]


def builder(conf: tuple, updated_conf: tuple = ()) -> DQV:
    constr, args = conf
    new_args = {}
    for k, v in args.items():
        if isinstance(v, tuple):
            instance, x = builder(v, updated_conf)
            print(f"received: {instance}, {x}")
            updated_conf += (v[0], x)
            print(f"updated_conf: {updated_conf}")
            # instance, new_args[k] = builder(v)
            new_args[k] = instance
        else:
            new_args[k] = v
    return constr.from_dict(new_args), updated_conf


c = (
    DQV,
    {
        "rng": (CustomRNG, {"seed": 5}),
        "policy_evaluator": (
            Egreedy,
            {"num_actions": 2, "epsilon_eval": 10},
        ),
    },
)

el, conf = builder(c)
