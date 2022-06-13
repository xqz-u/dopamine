import logging

import gin
from attrs import define
from thesis import types
from thesis.runner import base

logger = logging.getLogger(__name__)


@gin.configurable
@define
class OnlineRunner(base.Runner):
    record_experience: bool = False

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.record_experience:
            # avoid gc'ing every
            # `OutOfGraphReplayBuffer._checkpoint_duration`
            self.agent.memory._keep_every = 1
            logger.info("Set memory's `keep_every` param to 1")
            self.agent.memory.full_experience_initializer(
                self._checkpoint_dir, self.steps, self.iterations
            )
            logger.info(
                f"Save full experience to {self.agent.memory._full_experience_path}"
            )

    def train_iteration(self) -> types.MetricsDict:
        logger.info("START online training...")
        episodes_dict = {
            "reward": 0.0,
            "steps": 0,
            "episodes": 0,
            **self.agent.initial_train_dict,
        }
        while episodes_dict["steps"] < self.steps:
            episodes_dict = base.accumulate_metrics(
                episodes_dict, self.agent_env_loop("train"), self
            )
        self.global_steps += episodes_dict["steps"]
        summ_episodes_dict = base.summarise_metrics(episodes_dict, self)
        self.report_metrics(episodes_dict, summ_episodes_dict, "train")
        self.checkpoint_experiment()
        return {"train": {"raw": episodes_dict, "summary": summ_episodes_dict}}

    def finalize_experiment(self):
        # record pending transitions if want full experence
        if self.record_experience:
            logger.info("Finalizing experience record")
            self.agent.memory.finalize_full_experience()
        super().finalize_experiment()
