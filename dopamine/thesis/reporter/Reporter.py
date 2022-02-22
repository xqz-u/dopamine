from abc import ABC, abstractmethod

import attr


@attr.s(auto_attribs=True)
class Reporter(ABC):
    experiment_name: str
    writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def setup(self, params: dict, run_number: int):
        pass

    @abstractmethod
    def __call__(self, raw_reports: dict, agg_reports: dict, runner_info: dict):
        pass
