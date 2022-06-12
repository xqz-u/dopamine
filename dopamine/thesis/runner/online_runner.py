import logging

import gin
from attrs import define
from thesis import types
from thesis.runner import base

logger = logging.getLogger(__name__)


@gin.configurable
@define
class OnlineRunner(base.Runner):
    def train_iteration(self) -> types.MetricsDict:
        logger.info("START online training...")
        episodes_dict = {
            "reward": 0.0,
            "steps": 0,
            "episodes": 0,
            **self.agent.initial_train_dict,
        }
        while episodes_dict["steps"] < self.eval_steps:
            episodes_dict = base.accumulate_metrics(
                episodes_dict, self.agent_env_loop("train"), self
            )
        self.global_steps += episodes_dict["steps"]
        summ_episodes_dict = base.summarise_metrics(episodes_dict, self)
        self.report_metrics(episodes_dict, summ_episodes_dict, "train")
        self.checkpoint_experiment()
        return {"train": {"raw": episodes_dict, "summary": summ_episodes_dict}}
