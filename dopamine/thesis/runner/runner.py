import datetime
import multiprocessing as mp
import os
import time
from typing import List, Tuple, Union

from thesis.runner import offline_runner, online_runner


def train_iteration(self):
    self.agent.eval_mode = False
    self._run_episodes("train")


# schedule:
# - train (default)
# - continuous_train_and_eval
# rl_mode:
# - online (default)
# - offline
def create_runner(
    conf: dict,
) -> Union[online_runner.OnlineRunner, offline_runner.OfflineRunner]:
    # set some defaults
    for key, default in [["schedule", "train"], ["rl_mode", "online"]]:
        conf["runner"][key] = conf["runner"].get(key, default)
    # give correct runner
    if conf["runner"]["rl_mode"] == "online":
        if conf["runner"]["schedule"] == "train":
            online_runner.OnlineRunner.run_one_iteration = train_iteration
        return online_runner.OnlineRunner
    if conf["runner"]["schedule"] == "train":
        offline_runner.OfflineRunner.run_one_iteration = train_iteration
    return offline_runner.OfflineRunner


def mp_print(s: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}-{os.getpid()}-{s}")


# NOTE starting processes sequentially to avoid race conditions in sql
# for aim reporters
def run_experiment(args: Tuple[dict, int]):
    conf, wait_time = args
    time.sleep(wait_time)
    mp_print("starting...")
    manager = create_runner(conf)(conf, **conf["runner"]["experiment"])
    manager.run_experiment_with_redundancy()
    mp_print("done!")


def run_multiple_configs(configs: List[dict], n_workers: int = None):
    n_configs, cores = len(configs), mp.cpu_count()
    n_workers = n_workers or (n_configs if n_configs < cores else cores - 1)
    with mp.Pool(n_workers) as p:
        p.map(run_experiment, zip(configs, range(n_configs)))
