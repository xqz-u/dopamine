import multiprocessing as mp
from pprint import pprint
from typing import Any, List

from thesis import experiments, utils
from thesis.runner.base import Runner
from thesis.runner.fixed_batch_runner import FixedBatchRunner
from thesis.runner.online_runner import OnlineRunner


def run_experiment(conf: dict):
    runner_class = conf.pop("runner")
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


# TODO change previous experiments, they still pass runner_class and
# should instead be saved inside confs
def run_parallel(confs: List[dict]) -> List[Any]:
    with mp.Pool() as pool:
        res = pool.map(run_experiment, confs)
    return res
