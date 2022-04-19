import operator
from collections import OrderedDict

import attr
from thesis import utils
from thesis.runner import Runner


# NOTE the way the offline_circular_replay_buffer works, this runner
# will load a new set of replay buffers for each redundancy,
# corresponding to the ones recorded online; if
# FixedBatchRunner.redundancy is different than the nr. of
# redundancies used to generate the data online, the runner will fail
# as it is
@attr.s(auto_attribs=True)
class FixedBatchRunner(Runner.Runner):
    def __attrs_post_init__(self):
        # we are not collecting new experiences at all, so force
        # fitting immediately, and train at each step
        min_rep_hist = self.conf["agent"].get("min_replay_history")
        self.conf["agent"]["min_replay_history"] = 0
        train_freq = self.conf["agent"].get("train_freq")
        self.conf["agent"]["train_freq"] = 1
        super().__attrs_post_init__()
        if min_rep_hist is not None and min_rep_hist != 0:
            self.console.warning(
                f"Offline agents train immediately, got min_replay_history: {min_rep_hist}, now set to 0"
            )
        if train_freq is not None and train_freq != 1:
            self.console.warning(
                f"Offline agents fit trajectories at every step, got train_freq: {train_freq}, now set to 1"
            )
        if self.agent.clip_rewards:
            self.console.warning(
                "clip_rewards is True, so the recorded rewards should be already clipped"
            )

    def train_iteration(self) -> OrderedDict:
        train_info = OrderedDict(
            loss=self.agent.init_loss(), steps=self.steps, q_estimates=0.0
        )
        for _ in range(self.steps):
            # for i in range(self.steps):
            # print(f"Progress: {i}%", end="\r", flush=True)
            utils.inplace_dict_assoc(
                train_info, operator.add, update_dict=self.agent.learn()
            )
        self.global_steps += self.steps
        aggregate_info = super().aggregate_losses(self.agent.loss_names, train_info)
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    @property
    def console_name(self):
        return __name__
