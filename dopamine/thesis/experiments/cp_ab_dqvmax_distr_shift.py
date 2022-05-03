import os

from thesis import config
from thesis.experiments import pg_time_train_iter_cc
from thesis.memory import prio_offline_memory
from thesis.runner import runner


def buildconf(exp_name: str, env: str, version: str) -> dict:
    conf = pg_time_train_iter_cc.make_conf(exp_name)
    conf["env"] = config.make_env(env, version)
    conf["runner"]["experiment"] = {
        "schedule": "train_and_eval",
        "seed": 4,
        "steps": int(1e3),
        "iterations": 500,
        "eval_period": 2,
    }
    return conf


def buildconf_priority(exp_name: str, env: str, version: str) -> dict:
    conf = buildconf(exp_name, env, version)
    conf["memory"][
        "call_"
    ] = prio_offline_memory.PrioritizedOfflineOutOfGraphReplayBuffer
    return conf


def make_confs(
    exp_name: str,
    env: str,
    version: str,
    env_folder: str,
    data_basedir: str,
    conf_builder: callable = buildconf,
):
    return runner.expand_conf(
        conf_builder(exp_name, env, version),
        3,
        buffers_root_dir=os.path.join(
            data_basedir,
            f"{env}-{version}/DQNAgent/{env_folder}_dqn_full_experience_%%/checkpoints/full_experience",
        ),
    )


def classic_confs() -> list:
    return [
        make_confs("cp_dqvmax_distr_shift", "CartPole", "v1", "cp", config.data_dir),
        make_confs("ab_dqvmax_distr_shift", "Acrobot", "v1", "ab", config.data_dir),
    ]


def priority_confs_pg() -> list:
    return [
        make_confs(
            "cp_dqvmax_distr_shift_prio",
            "CartPole",
            "v1",
            "cp",
            config.peregrine_data_dir,
            conf_builder=buildconf_priority,
        ),
        make_confs(
            "ab_dqvmax_distr_shift_prio",
            "Acrobot",
            "v1",
            "ab",
            config.peregrine_data_dir,
            conf_builder=buildconf_priority,
        ),
    ]


if __name__ == "__main__":
    # confs = classic_confs()
    confs = priority_confs_pg()
    runner.p_run_experiments([c for c_env in confs for c in c_env])
