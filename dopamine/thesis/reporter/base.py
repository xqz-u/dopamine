from abc import ABC, abstractmethod

import gin
from attrs import define, field
from thesis import types


# TODO is experiment_name really mandatory? cant we let decide the
# values passed by the runner what is really necessary?
@gin.configurable
@define
class Reporter(ABC):
    experiment_name: str
    writing_freq: int = field(kw_only=True, default=1)

    # define a post init that subclasses will call if they override
    # their own post inits; in this particular case, do nothing
    def __attrs_post_init__(self):
        ...

    # NOTE current reporters only use summ_reports
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
