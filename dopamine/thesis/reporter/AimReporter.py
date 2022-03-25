import aim
import attr
from thesis import utils
from thesis.reporter import Reporter


# NOTE giving a Run a hash allows to resume it
# BUT hashing all arguments together as an iterable does not give
# the same hash, even when all the components are strings;
# investigate why, this manual hash should be enough as long as
# any of the components is systematically changed (usually
# experiment_name). Final solution would be to hash the whole dict
@attr.s(auto_attribs=True)
class AimReporter(Reporter.Reporter):
    repo: str
    conf: dict
    writer: aim.Run = attr.ib(init=False)

    @property
    def hparams(self) -> dict:
        r_conf = utils.reportable_conf(self.conf)
        for k in ["base_dir", "ckpt_file_prefix", "logging_file_prefix"]:
            r_conf["runner"].pop(k, None)
        r_conf["env"].pop("preproc", None)
        r_conf.pop("reporters", None)
        return r_conf

    def __attrs_post_init__(self):
        run_hash = "@".join(
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
        self.writer = aim.Run(
            run_hash=str(run_hash), repo=self.repo, experiment=self.experiment_name
        )
        self.conf["hash"] = run_hash
        self.writer["hparams"] = self.hparams

    def __call__(self, _, agg_reports: dict, runner_info: dict):
        for tag, val in agg_reports.items():
            self.writer.track(
                val,
                name=tag,
                step=runner_info["global_steps"],
                epoch=runner_info["curr_iteration"],
                context={"context": runner_info["current_schedule"]},
            )
