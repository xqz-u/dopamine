import logging

import gin
from attrs import define
from thesis import types
from thesis.runner import base

logger = logging.getLogger(__name__)


@gin.configurable
@define
class FixedBatchRunner(base.Runner):
    def train_iteration(self) -> types.MetricsDict:
        logger.info("START offline training...")
        episodes_dict = {"Steps": self.steps, **self.agent.initial_train_dict}
        for _ in range(self.steps):
            # NOTE bootstrap: problematic when a bad policy is picked.
            # online it will die out quick, here it is used for longer;
            # but this also means that it has more time to improve
            self.agent.on_episode_start("train")
            episodes_dict = self.agent.train_accumulate(
                episodes_dict, self.agent.learn()
            )
        self.global_steps += self.steps
        summ_metrics = base.summarise_metrics(episodes_dict, self)
        self.report_metrics(episodes_dict, summ_metrics, "train")
        self.checkpoint_experiment()
        return {"train": {"raw": episodes_dict, "summary": summ_metrics}}
