from abc import ABC, abstractmethod

import gin
from attrs import define, field
from thesis import types


@gin.configurable
@define
class Reporter(ABC):
    experiment_name: str
    writing_freq: int = field(kw_only=True, default=1)

    # define a post init that subclasses will call if they override
    # their own post inits; in this particular case, do nothing
    def __attrs_post_init__(self):
        ...

    @abstractmethod
    def __call__(
        self,
        raw_reports: types.MetricsDict,
        summ_reports: types.MetricsDict,
        experiment_info: dict,
    ):
        ...

    @abstractmethod
    def finalize(self):
        ...
