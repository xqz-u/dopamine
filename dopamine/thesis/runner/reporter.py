from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import aim
import attr
import numpy as np
from jax import numpy as jnp


@attr.s
class Reporter(ABC):
    writing_freq: int = attr.ib(default=1, kw_only=True)

    @abstractmethod
    def setup(self, iteration: int):
        pass

    @abstractmethod
    def __call__(
        self,
        reports: list,
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

    def setup(self, iteration: int):
        self.experiment = f"{self.experiment}_{iteration}"
        self.writer = aim.Run(repo=self.repo, experiment=self.experiment)

    def __call__(
        self,
        reports: List[Tuple[str, Union[np.ndarray, jnp.DeviceArray]]],
        step: int,
        epoch: int = None,
        context: dict = None,
    ):
        if step % self.writing_freq == 0:
            for tag, val in reports:
                self.writer.track(
                    float(val),
                    name=tag,
                    step=step,
                    epoch=epoch,
                    context=context,
                )
