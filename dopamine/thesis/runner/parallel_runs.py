import multiprocessing as mp
from typing import List

from thesis.runner import runner


def run_experiment(conf: dict):
    manager = runner.create_runner(conf)
    manager.run_experiment_with_redundancy()


# FIXME problem arises with Aim, I believe race condition on table
# creation...
def run_multiple_configs(configs: List[dict], n_workers: int = None):
    n_configs, cores = len(configs), mp.cpu_count()
    n_workers = n_workers or (n_configs if n_configs < cores else cores - 1)
    with mp.Pool(n_workers) as p:
        p.map(run_experiment, configs)
