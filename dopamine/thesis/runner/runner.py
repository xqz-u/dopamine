import datetime
import multiprocessing as mp
import os
import time
from copy import deepcopy
from typing import List, Tuple, Union

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


def expand_confs_repeats(confs_and_reps: List[Tuple[dict, int]]) -> List[dict]:
    def set_redundancy(conf: dict, rep: int) -> dict:
        c = deepcopy(conf)
        c["runner"]["experiment"]["redundancy_nr"] = rep
        return c

    ret = [
        set_redundancy(conf, i)
        for conf, repeat in confs_and_reps
        for i in range(repeat)
    ]
    print("Experiments running order:")
    for c in ret:
        print(c["experiment_name"], c["runner"]["experiment"]["redundancy_nr"])
    return ret


# check that conf.memory is offline, descend into dirs from
# _buffers_root_dir and add those to each config under the keyword
# required by the offline buffer
# TODO load offline logs correctly! in parallel too
# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment_atomic(conf: dict, init_wait: float = None):
    if init_wait is not None:
        mp_print(f"Sleep {init_wait}s...")
        time.sleep(init_wait)
    redundancy_nr = conf["runner"]["experiment"].get("redundancy_nr")
    assert isinstance(
        redundancy_nr, int
    ), f"redundancy_nr should be int, got {redundancy_nr}"
    mp_print(f"Start redundancy {redundancy_nr}")
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    run = create_runner(conf)
    run.run_experiment()
    mp_print("DONE!")


def run_experiment_atomic_wrap(args):
    return run_experiment_atomic(*args)


def run_experiments(confs_and_reps: List[Tuple[dict, int]]):
    for c in expand_confs_repeats(confs_and_reps):
        run_experiment_atomic(c)


# FIXME when SIGINT is given, something happens (the signal handler
# for mongo does not run); when the run is resumed, an error related
# to the parallel code occurs, usually one process gets stuck in a
# deadlock and the whole program never terminates. So don't start and
# stop at all right now!
def p_run_experiments(confs_and_reps: List[Tuple[dict, int]]):
    expanded_confs = expand_confs_repeats(confs_and_reps)
    n_confs, n_workers = os.cpu_count(), len(expanded_confs)
    n_workers = n_confs if n_confs < n_workers else n_workers
    # NOTE processes are spawned at time instants 0, 1, 2, ...,
    # n_confs-1
    with mp.Pool(processes=n_workers) as p:
        p.map(run_experiment_atomic_wrap, zip(expanded_confs, range(n_confs)))
