import aim
import attr
from thesis.reporter import Reporter


# NOTE giving a Run a hash allows to resume it
@attr.s(auto_attribs=True)
class AimReporter(Reporter.Reporter):
    repo: str
    writer: aim.Run = attr.ib(init=False)

    def setup(self, params: dict, run_number: int):
        run_hash = hash(
            tuple(
                [
                    self.experiment_name,
                    str(params["agent"]["call_"]),
                    f"{params['env']['environment_name']}-{params['env']['version']}",
                    str(run_number),
                ]
            )
        )
        self.writer = aim.Run(
            run_hash=str(run_hash), repo=self.repo, experiment=self.experiment_name
        )
        params["hash"] = run_hash
        self.writer["hparams"] = params

    def __call__(self, _, agg_reports: dict, runner_info: dict):
        for tag, val in agg_reports.items():
            self.writer.track(
                val,
                name=tag,
                step=runner_info["global_steps"],
                epoch=runner_info["curr_iteration"],
                context={"context": runner_info["current_schedule"]},
            )
