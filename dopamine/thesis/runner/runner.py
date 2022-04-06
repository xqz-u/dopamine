import datetime
import itertools as it
import multiprocessing as mp
import os
import time
from copy import deepcopy
from typing import List, Union

from thesis import config, utils
from thesis.memory import offline_memory
from thesis.runner.FixedBatchRunner import FixedBatchRunner
from thesis.runner.OnlineRunner import OnlineRunner


def mp_print(s: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}-{os.getpid()}-{s}")


def show_experiments_order(confs: List[dict]):
    print("Experiments running order:")
    for c in confs:
        print(c["experiment_name"], c["runner"]["experiment"]["redundancy_nr"])


# schedule:
# - train (default)
# - eval
# - train_and_eval
def create_runner(conf: dict) -> Union[OnlineRunner, FixedBatchRunner]:
    # set some defaults
    conf["runner"]["call_"] = conf["runner"].get("call_", OnlineRunner)
    return conf["runner"]["call_"](conf, **conf["runner"]["experiment"])


def add_redundancies(conf: dict, repeat: int) -> List[dict]:
    expanded_confs = []
    for i in range(repeat):
        c = deepcopy(conf)
        c["runner"]["experiment"]["redundancy_nr"] = i
        expanded_confs.append(c)
    return expanded_confs


# NOTE when repeats is > than available redundancy dirs/specified
# iterations, the latter  are cycled through for a total of repeats
# times, in all other cases repeats takes the priority
def add_offline_buffers(
    confs: List[dict],
    buffers_root_dir: str,
    intermediate_dirs: str = "",
    iterations: List[List[int]] = None,
) -> List[dict]:
    for c, buff_dir, iters_ in zip(
        confs,
        it.cycle(utils.unfold_replay_buffers_dir(buffers_root_dir, intermediate_dirs)),
        it.cycle(iterations or ([None] * len(confs))),
    ):
        assert (
            c["memory"].get("call_") is offline_memory.OfflineOutOfGraphReplayBuffer
        ), f"buffers_root_dir is {buffers_root_dir}, so conf['memory']['call_'] should be {offline_memory.OfflineOutOfGraphReplayBuffer}"
        c["memory"]["_buffers_dir"] = buff_dir
        if iters_ is not None:
            c["memory"]["_buffers_iterations"] = iters_
    return confs


# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment_atomic(conf: dict, scratch: bool, init_wait: float = None):
    if init_wait is not None:
        mp_print(f"Sleep {init_wait}s...")
        time.sleep(init_wait)
    redundancy_nr = conf["runner"]["experiment"].get("redundancy_nr")
    assert isinstance(
        redundancy_nr, int
    ), f"redundancy_nr should be int, got {redundancy_nr}"
    mp_print(f"Start redundancy {redundancy_nr}")
    utils.data_dir_from_conf(
        conf["experiment_name"],
        conf,
        basedir=config.scratch_data_dir if scratch else None,
    )
    run = create_runner(conf)
    run.run_experiment()
    mp_print("DONE!")


def run_experiments(experiments_confs: List[dict], scratch: bool = False):
    show_experiments_order(experiments_confs)
    for c in experiments_confs:
        run_experiment_atomic(c, scratch)


# FIXME when SIGINT is given, something happens (the signal handler
# for mongo does not run); when the run is resumed, an error related
# to the parallel code occurs, usually one process gets stuck in a
# deadlock and the whole program never terminates. So don't start and
# stop at all right now!
def p_run_experiments(experiments_confs: List[dict], scratch: bool = False):
    show_experiments_order(experiments_confs)
    n_confs, n_workers = os.cpu_count(), len(experiments_confs)
    n_workers = n_confs if n_confs < n_workers else n_workers
    # NOTE processes are spawned at time instants 0, 1, 2, ...,
    # n_confs-1
    with mp.Pool(processes=n_workers) as p:
        p.starmap(
            run_experiment_atomic,
            zip(experiments_confs, [scratch] * n_confs, range(n_confs)),
        )
