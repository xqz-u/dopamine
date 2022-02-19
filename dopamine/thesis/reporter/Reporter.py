from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import attr


@attr.s(auto_attribs=True)
class Reporter(ABC):
    experiment_name: str
    writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def setup(self, params: dict, run_number: int):
        pass

    @abstractmethod
    def __call__(
        self, reports: dict, step: int, epoch: int = None, context: dict = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        pass
