import datetime
import itertools as it
import multiprocessing as mp
import os
import time
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from thesis import offline_circular_replay_buffer, utils
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


# NOTE when repeats is > than available redundancy dirs, the latter
# are cycled through for a total of repeats times, in all other cases
# repeats takes the priority
def expand_single_conf(
    conf: dict,
    repeats: int,
    buffers_root_dir: str = None,
    intermediate_dirs: str = "",
) -> List[dict]:
    expanded_confs = []
    for i in range(repeats):
        c = deepcopy(conf)
        c["runner"]["experiment"]["redundancy_nr"] = i
        expanded_confs.append(c)
    if buffers_root_dir is not None:
        assert (
            conf["memory"].get("call_")
            is offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer
        ), f"buffers_root_dir is {buffers_root_dir}, so conf['memory']['call_'] should be {offline_circular_replay_buffer.OfflineOutOfGraphReplayBuffer}"
        for c, buff_dir in zip(
            expanded_confs,
            it.cycle(
                utils.unfold_replay_buffers_dir(buffers_root_dir, intermediate_dirs)
            ),
        ):
            c["memory"]["_buffers_dir"] = buff_dir
    return expanded_confs


def expand_configs(
    experiments_specs: Tuple[dict, int, Optional[str], Optional[str]]
) -> List[dict]:
    expanded_confs = [
        c for args in experiments_specs for c in expand_single_conf(*args)
    ]
    print("Experiments running order:")
    for c in expanded_confs:
        print(c["experiment_name"], c["runner"]["experiment"]["redundancy_nr"])
    return expanded_confs


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


def run_experiments(experiments_specs: Tuple[dict, int, Optional[str], Optional[str]]):
    for c in expand_configs(experiments_specs):
        run_experiment_atomic(c)


# FIXME when SIGINT is given, something happens (the signal handler
# for mongo does not run); when the run is resumed, an error related
# to the parallel code occurs, usually one process gets stuck in a
# deadlock and the whole program never terminates. So don't start and
# stop at all right now!
def p_run_experiments(
    experiments_specs: Tuple[dict, int, Optional[str], Optional[str]]
):
    expanded_confs = expand_configs(experiments_specs)
    n_confs, n_workers = os.cpu_count(), len(expanded_confs)
    n_workers = n_confs if n_confs < n_workers else n_workers
    # NOTE processes are spawned at time instants 0, 1, 2, ...,
    # n_confs-1
    with mp.Pool(processes=n_workers) as p:
        p.map(run_experiment_atomic_wrap, zip(expanded_confs, range(n_confs)))
