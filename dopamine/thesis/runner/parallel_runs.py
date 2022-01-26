import datetime
import multiprocessing as mp
import os
import time
from typing import List, Tuple

from thesis.runner import runner


def mp_print(s: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}-{os.getpid()}-{s}")


# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment(args: Tuple[dict, int]):
    conf, wait_time = args
    time.sleep(wait_time)
    mp_print("starting...")
    manager = runner.create_runner(conf)
    manager.run_experiment_with_redundancy()
    mp_print("done!")


def run_multiple_configs(configs: List[dict], n_workers: int = None):
    n_configs, cores = len(configs), mp.cpu_count()
    n_workers = n_workers or (n_configs if n_configs < cores else cores - 1)
    with mp.Pool(n_workers) as p:
        p.map(run_experiment, zip(configs, range(n_configs)))
