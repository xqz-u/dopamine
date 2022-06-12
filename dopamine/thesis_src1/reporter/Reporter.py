from abc import ABC, abstractmethod

import attr


# NOTE giving a Run a hash allows to resume it in Aim, and to
# differintiate on runs if anything goes wrong
# BUT hashing all arguments together as an iterable does not give
# the same hash, even when all the components are strings;
# investigate why, this manual hash should be enough as long as
# any of the components is systematically changed (usually
# experiment_name). Final solution would be to hash the whole config
# dict
@attr.s(auto_attribs=True)
class Reporter(ABC):
    experiment_name: str
    conf: dict
    writing_freq: int = attr.ib(default=1, kw_only=True)

    def __attrs_post_init__(self):
        self.conf["run_hash"] = "@".join(
            [
                str(x)
                for x in (
                    self.conf["experiment_name"],
                    self.conf["agent"]["call_"],
                    self.conf["env"]["environment_name"],
                    self.conf["env"]["version"],
                    self.conf["runner"]["experiment"]["redundancy_nr"],
                )
            ]
        )

    @abstractmethod
    def __call__(self, raw_reports: dict, agg_reports: dict, runner_info: dict):
        pass
