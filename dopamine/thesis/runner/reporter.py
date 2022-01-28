from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import aim
import attr


@attr.s
class Reporter(ABC):
    writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def setup(self, params: dict, run_number: int):
        pass

    @abstractmethod
    def __call__(
        self,
        reports: dict,
        step: int,
        epoch: int = None,
        context: dict = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        pass


# NOTE giving a Run a hash allows to resume it
@attr.s(auto_attribs=True)
class AimReporter(Reporter):
    repo: str
    experiment: str
    writer: aim.Run = attr.ib(init=False)

    def setup(self, params: dict, run_number: int):
        exp_name = f"{self.experiment}_{run_number}"
        self.writer = aim.Run(run_hash=exp_name, repo=self.repo, experiment=exp_name)
        self.writer["hparams"] = params

    def __call__(
        self,
        reports: Dict[str, Union[float, Dict[str, float]]],
        step: int,
        epoch: int = None,
        context: dict = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
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
        return {"aim_reports": agg_reports}
