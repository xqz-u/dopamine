import logging

import aim
import gin
from attrs import define, field
from thesis import types
from thesis.reporter import base

logger = logging.getLogger(__name__)


# NOTE if aim.Run is given a `run_hash`, aim is able to resume the run.
# useful for start and stop
@gin.configurable
@define
class AimReporter(base.Reporter):
    repo: str
    writer: aim.Run = field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.writer = aim.Run(
            repo=self.repo,
            experiment=self.experiment_name,
        )
        logger.info(f"Initialized AIM repo at: {self.writer.repo.path}")

    def __call__(self, _, summ_reports: types.MetricsDict, experiment_info: dict):
        for tag, val in summ_reports.items():
            self.writer.track(
                val,
                name=tag,
                step=experiment_info["Global_steps"],
                epoch=experiment_info["Iteration"],
                context={"context": experiment_info["Schedule"]},
            )
        logger.debug("Sent summarised reports to AIM")

    # no real finalization needed here
    def finalize(self):
        ...

    def register_conf(self, conf: dict):
        logger.info("Registering experiment config in Aim...")
        self.writer["hparams"] = conf
