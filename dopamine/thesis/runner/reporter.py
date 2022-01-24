from abc import ABC, abstractmethod
from typing import Dict, Union

import aim
import attr


@attr.s
class Reporter(ABC):
    writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def setup(self, iteration: int):
        pass

    @abstractmethod
    def __call__(
        self,
        reports: dict,
        step: int,
        epoch: int = None,
        context: dict = None,
    ):
        pass


# NOTE it would be ideal to extend aim.Run too, but inheritance will not
# play nicely since it is not attr'd (you can still attr a class you do
# not own, but need to specify the fields which should be attr.ib'd)
@attr.s(auto_attribs=True)
class AimReporter(Reporter):
    repo: str
    experiment: str
    writer: aim.Run = attr.ib(init=False)

    def setup(self, iteration: int, params: dict):
        self.experiment = f"{self.experiment}_{iteration}"
        self.writer = aim.Run(repo=self.repo, experiment=self.experiment)
        self.writer["hparams"] = params

    def __call__(
        self,
        reports: Dict[str, Union[float, Dict[str, float]]],
        step: int,
        epoch: int = None,
        context: dict = None,
    ):
        if step % self.writing_freq:
            return
        losses = reports["losses"]
        agg_reports = [
            ("AvgEp_return", reports["return"] / reports["episodes"])
        ] + list(
            zip(
                map(lambda t: f"AvgEp_{t}", losses.keys()),
                [v / reports["loss_steps"] for v in losses.values()],
            )
        )
        for tag, val in agg_reports:
            self.writer.track(
                val,
                name=tag,
                step=step,
                epoch=epoch,
                context=context,
            )
