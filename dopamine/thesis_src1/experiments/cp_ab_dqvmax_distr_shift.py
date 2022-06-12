import os

from thesis import config, constants
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


def classic_confs(datadir) -> list:
    return [
        make_confs("cp_dqvmax_distr_shift_baseline", "CartPole", "v1", "cp", datadir),
        make_confs("ab_dqvmax_distr_shift_baseline", "Acrobot", "v1", "ab", datadir),
    ]


def priority_confs(datadir) -> list:
    return [
        make_confs(
            "cp_dqvmax_distr_shift_prio",
            "CartPole",
            "v1",
            "cp",
            datadir,
            conf_builder=buildconf_priority,
        ),
        make_confs(
            "ab_dqvmax_distr_shift_prio",
            "Acrobot",
            "v1",
            "ab",
            datadir,
            conf_builder=buildconf_priority,
        ),
    ]


if __name__ == "__main__":
    logs_dir = constants.data_dir
    # logs_dir = constants.peregrine_data_dir
    confs = classic_confs(logs_dir)
    # confs = priority_confs(logs_dir)
    confs = [c for c_env in confs for c in c_env]
    runner.p_run_experiments(confs, logs_dir=logs_dir)
