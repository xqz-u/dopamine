import aim
import attr
from thesis import utils
from thesis.reporter import Reporter


@attr.s(auto_attribs=True)
class AimReporter(Reporter.Reporter):
    repo: str
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
        super().__attrs_post_init__()
        self.writer = aim.Run(
            run_hash=str(self.conf["run_hash"]),
            repo=self.repo,
            experiment=self.experiment_name,
        )
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
