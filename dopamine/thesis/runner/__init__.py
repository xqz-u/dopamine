import multiprocessing as mp
from pprint import pprint
from typing import Any, List

from thesis import experiments, utils
from thesis.runner.base import Runner
from thesis.runner.fixed_batch_runner import FixedBatchRunner
from thesis.runner.online_runner import OnlineRunner


def run_experiment(conf: dict, runner_class: Runner):
    run = runner_class(**experiments.make_conf(**conf))
    c = utils.reportable_config(
        {
            "call_": utils.callable_name_getter(run),
            **utils.config_collector(run, "reportable"),
        }
    )
    pprint(c)
    for rep in run.reporters:
        rep.register_conf(c)
    run.run()


# return results to caller for completion - there won't be any here
def run_parallel(confs_subsets: List[dict], runner_class: Runner) -> List[Any]:
    with mp.Pool() as pool:
        res = pool.starmap(
            run_experiment,
            zip(confs_subsets, [runner_class] * len(confs_subsets)),
        )
    return res
