import logging
import os
import pprint
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List, Tuple, Union

import gin
import jax
import numpy as np
from attrs import define, field
from dopamine.discrete_domains import atari_lib, checkpointer, gym_lib
from thesis import agent, reporter, types, utils

logger = logging.getLogger(__name__)


# TODO instantiation! can use absl.flags or should use a dict config?
#      this goes together with Config! + collection
# TODO update git upstream and propose fix to CircularReplayBuffer
# TODO update project dependencies
# TODO change keys of metrics in order to avoid processing them in R?
#      report summarised metrics to mongo to avoid doing them again
#      in R?
# TODO simplify offline_memory.OfflineOutOfGraphReplayBuffer?
# NOTE @define has slots=True, whitch prevents runtime monkeypatching
# (see
# https://www.attrs.org/en/stable/glossary.html#term-slotted-classes).
# for this reason, no mixins are run after __attrs_post_init__ rn, since
# this limits the mixins' scope; switch to dataclasses or use
# slots=False to fix
@gin.configurable
@define
class Runner(ABC):
    agent: agent.Agent
    checkpoint_base_dir: str
    env: Union[gym_lib.GymPreprocessing, atari_lib.AtariPreprocessing]
    experiment_name: str
    iterations: int
    steps: int
    eval_period: int = 5
    eval_steps: int = 500
    # list of callbacks which take in the dictionary of metrics
    # collected by the runner in the agent-env interaction loop, and a
    # dict of additional information returned by a PolicyEvaluator
    # e.g. Egreedy; these callbacks run at every action selection
    # step, when they returned a the original dict enriched with by
    # their own logic
    on_policy_eval: List[
        Callable[[types.MetricsDict, types.MetricsDict], types.MetricsDict]
    ] = field(factory=list)
    redundancy: int = 0
    reporters: List[reporter.Reporter] = field(factory=list)
    schedule: str = "train_and_eval"
    curr_iteration: int = field(init=False, default=0)
    global_steps: int = field(init=False, default=0)
    _checkpoint_dir: str = field(init=False)
    _checkpointer: checkpointer.Checkpointer = field(init=False)

    def __attrs_post_init__(self):
        self._checkpoint_dir = os.path.join(
            str(self.checkpoint_base_dir), self.experiment_name, str(self.redundancy)
        )
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir)
        logger.info(f"Created checkpointer at: {self._checkpointer._base_directory}")
        logger.info("Configured reporters:")
        for rep in self.reporters:
            logger.info(rep)

    @abstractmethod
    def train_iteration(self) -> types.MetricsDict:
        ...

    def run(self):
        self.ensure_reproducible()
        self.curr_iteration, self.global_steps = 0, 0
        while self.curr_iteration < self.iterations:
            metrics = getattr(self, f"{self.schedule}_iteration")()
            logger.info(
                f"DONE Iteration {self.curr_iteration}\n{pprint.pformat(metrics)}"
            )
            self.curr_iteration += 1
        logger.info("Finalize reporters...")
        for reporter in self.reporters:
            reporter.finalize()

    def step_environment(
        self, action: int, episode_steps: int
    ) -> Tuple[np.array, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        done = done or episode_steps >= self.env.environment.spec.max_episode_steps
        return observation, reward, done, info

    def agent_env_loop(self, mode: str) -> types.MetricsDict:
        evaluation = mode == "eval"
        episode_dict = {
            "reward": 0.0,
            "steps": 0,
            **({} if evaluation else self.agent.initial_train_dict),
        }
        obs, done = self.env.reset(), False
        while not done:
            action, more_info = self.agent.select_action(obs, mode)
            obs, reward, done, _ = self.step_environment(action, episode_dict["steps"])
            for cb in self.on_policy_eval:
                episode_dict = cb(episode_dict, more_info)
            if not evaluation:
                self.agent.record_trajectory(reward, done)
                episode_dict = self.agent.train_accumulate(
                    episode_dict, self.agent.learn()
                )
            episode_dict["reward"] += reward
            episode_dict["steps"] += 1
        return episode_dict

    # NOTE default implementation, can be overridden like
    # self.train_iteration; no checkpointing performed here
    def eval_iteration(self) -> types.MetricsDict:
        logger.info("START evaluation...")
        episodes_dict = {"reward": 0.0, "steps": 0, "episodes": 0}
        while episodes_dict["steps"] < self.eval_steps:
            episode_stats = self.agent_env_loop("eval")
            logger.debug(f"\tEval episode: {episode_stats}")
            episodes_dict = accumulate_metrics(episodes_dict, episode_stats, self)
        # during training, training steps take the precedence; same when
        # training and evaluation are interleaved
        if self.schedule == "eval":
            self.global_steps += episodes_dict["steps"]
        summ_episodes_dict = summarise_metrics(episodes_dict, self)
        self.report_metrics(episodes_dict, summ_episodes_dict, "eval")
        return {"eval": {"raw": episodes_dict, "summary": summ_episodes_dict}}

    def train_and_eval_iteration(self) -> types.MetricsDict:
        metrics = self.train_iteration()
        if not self.curr_iteration % self.eval_period:
            return {**metrics, **self.eval_iteration()}
        return metrics

    def ensure_reproducible(self):
        # experiment's seed is determined by the agent's RNG
        # - cast to int, it is converted to jnp.ndarray as a PyTree
        seed = int(self.agent.rng.seed)
        logger.info(f"EXPERIMENT SEED: {seed}")
        # set numpy's global rng state for Dopamine code (I do not use
        # numpy's random API)
        np.random.seed(seed)
        logger.info("Seeded numpy's RNG")
        # set gym's state
        self.env.environment.reset(seed=seed)
        logger.info("Seeded gym's RNG")

    def report_metrics(
        self, metrics: types.MetricsDict, metrics_summary: types.MetricsDict, mode: str
    ):
        for reporter in self.reporters:
            reporter(
                metrics,
                metrics_summary,
                {
                    "agent": self.agent.name,
                    "env_name": self.env.environment.spec.name,
                    "env_version": self.env.environment.spec.version,
                    "iteration": self.curr_iteration,
                    "redundancy": self.redundancy,
                    "global_steps": self.global_steps,
                    "schedule": mode,
                },
            )

    # NOTE checkpoints are taken only on training iterations, where the
    # memory buffer grows and the agent's parameters are updated
    def checkpoint_experiment(self):
        # checkpoint agent state
        self._checkpointer.save_checkpoint(self.curr_iteration, self.agent.serializable)
        # checkpoint memory
        self.agent.memory.save(self._checkpoint_dir, self.curr_iteration)
        logger.debug(
            f"Wrote memory checkpoint #{self.curr_iteration} at {self._checkpoint_dir}"
        )


def accumulate_metrics(
    acc: types.MetricsDict, episode_dict: types.MetricsDict, runner: Runner
) -> types.MetricsDict:
    if "loss" in episode_dict:
        acc = runner.agent.train_accumulate(acc, episode_dict)
    acc["reward"] += episode_dict["reward"]
    acc["steps"] += episode_dict["steps"]
    acc["episodes"] += 1
    if "max_q_s0" in episode_dict:
        acc["max_q_s0"] = acc.get("max_q_s0", 0) + episode_dict["max_q_s0"]
    return acc


def summarise_metrics(
    episodes_dict: types.MetricsDict, runner: Runner
) -> types.MetricsDict:
    summ_dict = deepcopy(episodes_dict)
    if "loss" in episodes_dict:
        episodes_dict["loss"] = jax.tree_map(float, episodes_dict["loss"])
        for loss_name, loss in episodes_dict["loss"].items():
            summ_dict[f"{loss_name}-Loss"] = loss / summ_dict["steps"]
        del summ_dict["loss"]
    if "reward" in summ_dict:
        summ_dict["reward"] /= summ_dict["episodes"]
    if "max_q_s0" in summ_dict:
        episodes_dict["max_q_s0"] = float(episodes_dict["max_q_s0"])
        summ_dict["max_q_s0"] = float(summ_dict["max_q_s0"])
        env_optimal_q_s0 = utils.deterministic_discounted_return(
            runner.env.environment, runner.agent.gamma
        )
        summ_dict["max_q_s0"] /= summ_dict["episodes"]
        summ_dict["qstar_s0"] = env_optimal_q_s0
        episodes_dict["qstar_s0"] = env_optimal_q_s0
    return summ_dict
