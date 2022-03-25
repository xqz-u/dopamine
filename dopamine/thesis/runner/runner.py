import datetime
import multiprocessing as mp
import os
import time
from copy import deepcopy
from typing import Sequence, Union

from thesis import utils
from thesis.runner.FixedBatchRunner import FixedBatchRunner
from thesis.runner.OnlineRunner import OnlineRunner


def mp_print(s: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}-{os.getpid()}-{s}")


# schedule:
# - train (default)
# - eval
# - train_and_eval
def create_runner(
    conf: dict,
) -> Union[OnlineRunner, FixedBatchRunner]:
    # set some defaults
    conf["runner"]["call_"] = conf["runner"].get("call_", OnlineRunner)
    return conf["runner"]["call_"](conf, **conf["runner"]["experiment"])


# TODO load offline logs correctly! also in parallel
# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment_atomic(conf: dict, redundancy_nr: int = 0, init_wait: float = None):
    if init_wait is not None:
        mp_print(f"Sleep {init_wait}s...")
        time.sleep(init_wait)
    mp_print(f"Start redundancy {redundancy_nr}")
    conf["runner"]["experiment"]["redundancy_nr"] = redundancy_nr
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    run = create_runner(conf)
    run.run_experiment()
    mp_print("DONE!")


def run_experiment_atomic_wrap(args):
    return run_experiment_atomic(*args)


def run_experiment(conf: dict, repeat: int = 1):
    for i in range(repeat):
        # the config is modified by the runner, so keep it as it is...
        run_experiment_atomic(deepcopy(conf), i)


# NOTE if repeat == n, then processes will be spawned at time instants
# 0, 1, 2, ..., n-1; if this is not enough, idea: give a delay in
# seconds which will be repeated for every redundancy
def p_run_experiment(conf: dict, repeat: int = 1, n_workers: int = None):
    n_workers = n_workers or (repeat if (cores := mp.cpu_count()) >= repeat else cores)
    print(f"Spawn pool of {n_workers} processes...")
    with mp.Pool(processes=n_workers) as pool:
        pool.map(
            run_experiment_atomic_wrap,
            zip([conf] * repeat, range(repeat), range(repeat)),
        )
    mp_print(f"DONE {repeat} redundancies")


def p_run_experiment_wrap(args):
    return p_run_experiment(*args)


def p_run_multiple_configs(
    configs: Sequence[dict], repeat: Sequence[int] = None, n_workers: int = None
):
    n_configs, cores = len(configs), mp.cpu_count()
    if repeat is None:
        repeat = [1] * n_configs
    else:
        assert len(repeat) == n_configs
    n_workers = n_workers or (n_configs if n_configs < cores else cores)
    print(f"n_workers: {n_workers}")
    # prefer running each config in parallel and its redundancies
    # serially, that's the reason of the [1] * n_configs
    with mp.Pool(n_workers) as p:
        p.map(p_run_experiment_wrap, zip(configs, repeat, [1] * n_configs))
