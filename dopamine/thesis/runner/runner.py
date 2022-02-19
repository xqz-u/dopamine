import datetime
import multiprocessing as mp
import os
import time
from typing import List, Tuple, Union

from thesis import utils
from thesis.runner.FixedBatchRunner import FixedBatchRunner
from thesis.runner.GrowingBatchRunner import GrowingBatchRunner
from thesis.runner.OnlineRunner import OnlineRunner


# schedule:
# - train (default)
# - eval
def create_runner(
    conf: dict,
) -> Union[OnlineRunner, GrowingBatchRunner, FixedBatchRunner]:
    # set some defaults
    for key, default in [["schedule", "train"], ["call_", OnlineRunner]]:
        conf["runner"][key] = conf["runner"].get(key, default)
    return conf["runner"]["call_"](conf, **conf["runner"]["experiment"])


def mp_print(s: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}-{os.getpid()}-{s}")


# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment(args: Tuple[dict, int]):
    conf, wait_time = args
    time.sleep(wait_time)
    mp_print("starting...")
    utils.data_dir_from_conf(conf["experiment_name"], conf)
    manager = create_runner(conf)
    manager.run_experiment_with_redundancy()
    mp_print("done!")


# TODO redundancies in parallel
def run_multiple_configs(configs: List[dict], n_workers: int = None):
    n_configs, cores = len(configs), mp.cpu_count()
    n_workers = n_workers or (n_configs if n_configs < cores else cores)
    print(f"n_workers: {n_workers}")
    with mp.Pool(n_workers) as p:
        p.map(run_experiment, zip(configs, range(n_configs)))
