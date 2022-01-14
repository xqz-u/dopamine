from abc import ABC, abstractmethod
from typing import Any, List

import aim
import attr


@attr.s
class Reporter(ABC):
    writer: Any = attr.ib(init=False)
    summary_writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def __call__(self, reports: list, i: int, **kwargs):
        pass


@attr.s(auto_attribs=True)
class AimReporter(Reporter):
    repo_path: str
    experiment_name: str

    def __post_init__(self):
        self.writer = aim.Run(repo=self.repo_path, experiment=self.experiment_name)

    def __call__(self, reports: List[dict], step: int, **_):
        if step % self.summary_writing_freq == 0:
            for rep in reports:
                self.writer.track(**rep, step=step)
