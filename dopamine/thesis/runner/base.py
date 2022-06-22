import logging
import os
import pprint
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List, Tuple

import jax
import numpy as np
from attrs import define, field
from dopamine.discrete_domains import checkpointer
from thesis import agent, reporter, types, utils

logger = logging.getLogger(__name__)


# NOTE @define has slots=True, whitch prevents runtime monkeypatching
# (see
# https://www.attrs.org/en/stable/glossary.html#term-slotted-classes).
# for this reason, no mixins are run after __attrs_post_init__ rn, since
# this limits the mixins' scope; switch to dataclasses or use
# slots=False to fix
@define
class Runner(ABC):
    agent: agent.Agent
    env: types.DiscreteEnv
    experiment_name: str
    checkpoint_base_dir: str
    iterations: int
    steps: int
    eval_period: int = 10
    eval_steps: int = 500
    # list of callbacks which take in the dictionary of metrics
    # collected by the runner in the agent-env interaction loop, and a
    # dict of additional information returned by a PolicyEvaluator
    # e.g. Egreedy, plus a string specifying training/evaluation;
    # these callbacks run at every action selection step, when they
    # returned a the original dict enriched with by their own logic
    on_policy_eval: List[
        Callable[[types.MetricsDict, types.MetricsDict, str], types.MetricsDict]
    ] = field(factory=list)
    redundancy: int = 0
    reporters: List[reporter.Reporter] = field(factory=list)
    schedule: str = "train_and_eval"
    curr_iteration: int = field(init=False, default=0)
    global_steps: int = field(init=False, default=0)
    env_name: str = field(init=False)
    _checkpointer: checkpointer.Checkpointer = field(init=False)
    _checkpoint_dir: str = field(init=False)

    def __attrs_post_init__(self):
        self._checkpoint_dir = os.path.join(
            self.checkpoint_base_dir, str(self.redundancy)
        )
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir)
        logger.info(f"Created checkpointer at: {self._checkpointer._base_directory}")
        logger.info("Configured reporters:")
        for rep in self.reporters:
            logger.info(rep)
        self.env_name = f"{self.env.spec.name}-v{self.env.spec.version}"

    @abstractmethod
    def train_iteration(self) -> types.MetricsDict:
        ...

    def run(self):
        self.ensure_reproducible()
        self.curr_iteration, self.global_steps = 0, 0
        while self.curr_iteration < self.iterations:
            metrics = getattr(self, f"{self.schedule}_iteration")()
            logger.info(
                f"DONE Iteration {self.curr_iteration}\nName: {self.experiment_name} Agent: {self.agent.name} Environment: {self.env_name}\n{pprint.pformat(metrics)}\n\n"
            )
            self.curr_iteration += 1
        self.finalize_experiment()

    def agent_env_loop(self, mode: str) -> types.MetricsDict:
        episode_dict = {
            "Reward": 0.0,
            "Steps": 0,
            **({} if mode == "eval" else self.agent.initial_train_dict),
        }
        obs, done = self.env.reset(), False
        while not done:
            action, more_info = self.agent.select_action(obs, mode)
            obs, reward, done, _ = self.env.step(action)
            for cb in self.on_policy_eval:
                episode_dict = cb(episode_dict, more_info, mode)
            if mode == "train":
                self.agent.record_trajectory(reward, done)
                episode_dict = self.agent.train_accumulate(
                    episode_dict, self.agent.learn()
                )
            episode_dict["Reward"] += reward
            episode_dict["Steps"] += 1
        return episode_dict

    # NOTE default implementation, can be overridden like
    # self.train_iteration; no checkpointing performed here
    def eval_iteration(self) -> types.MetricsDict:
        logger.info("START evaluation...")
        episodes_dict = {"Reward": 0.0, "Steps": 0, "Episodes": 0}
        while episodes_dict["Steps"] < self.eval_steps:
            episode_stats = self.agent_env_loop("eval")
            logger.debug(f"\tEval episode: {episode_stats}")
            episodes_dict = accumulate_metrics(episodes_dict, episode_stats, self)
        # during training, training steps take the precedence; same when
        # training and evaluation are interleaved
        if self.schedule == "eval":
            self.global_steps += episodes_dict["Steps"]
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
        self.env.reset(seed=seed)
        logger.info("Seeded gym's RNG")

    def report_metrics(
        self, metrics: types.MetricsDict, metrics_summary: types.MetricsDict, mode: str
    ):
        for reporter in self.reporters:
            reporter(
                metrics,
                metrics_summary,
                {
                    "Agent": self.agent.name,
                    "Env": self.env_name,
                    "Iteration": self.curr_iteration,
                    "Redundancy": self.redundancy,
                    "Global_steps": self.global_steps,
                    "Schedule": mode,
                },
            )

    # generally, additional hooks to run at the end of a run
    def finalize_experiment(self):
        logger.info("Running finalize hooks")
        for reporter in self.reporters:
            reporter.finalize()

    # NOTE checkpoints are taken only on training iterations, when the
    # agent parameters are updated
    # NOTE checkpoint memory when OnlineRunner.record_experience; in
    # other cases, checkpoints are only useful for start/stop
    # functionality, not implemented now
    def checkpoint_experiment(self):
        # checkpoint agent state
        logger.debug(f"Write agent checkpoint at {self._checkpoint_dir}")
        self._checkpointer.save_checkpoint(self.curr_iteration, self.agent.serializable)

    @property
    def reportable(self) -> Tuple[str]:
        return (
            "agent",
            "_checkpoint_dir",
            "experiment_name",
            "iterations",
            "steps",
            "eval_period",
            "eval_steps",
            "redundancy",
            "schedule",
            "env_name",
            (
                "on_policy_eval",
                lambda: [utils.callable_name_getter(c) for c in self.on_policy_eval],
            ),
        )


def accumulate_metrics(
    acc: types.MetricsDict, episode_dict: types.MetricsDict, runner: Runner
) -> types.MetricsDict:
    if "loss" in episode_dict:
        acc = runner.agent.train_accumulate(acc, episode_dict)
    acc["Reward"] += episode_dict["Reward"]
    acc["Steps"] += episode_dict["Steps"]
    acc["Episodes"] += 1
    if "Max_Q_S0" in episode_dict:
        acc["Max_Q_S0"] = acc.get("Max_Q_S0", 0) + episode_dict["Max_Q_S0"]
    return acc


def summarise_metrics(
    episodes_dict: types.MetricsDict, runner: Runner
) -> types.MetricsDict:
    summ_dict = deepcopy(episodes_dict)
    if "loss" in episodes_dict:
        episodes_dict["loss"] = jax.tree_map(float, episodes_dict["loss"])
        for loss_name, loss in episodes_dict["loss"].items():
            summ_dict[f"{loss_name}-Loss"] = loss / summ_dict["Steps"]
        del summ_dict["loss"]
    if "Reward" in summ_dict:
        summ_dict["Reward"] /= summ_dict["Episodes"]
    if "Max_Q_S0" in summ_dict:
        episodes_dict["Max_Q_S0"] = float(episodes_dict["Max_Q_S0"])
        summ_dict["Max_Q_S0"] = float(summ_dict["Max_Q_S0"])
        env_optimal_q_s0 = utils.deterministic_discounted_return(
            runner.env, runner.agent.gamma
        )
        summ_dict["Max_Q_S0"] /= summ_dict["Episodes"]
        summ_dict["QStar_S0"] = env_optimal_q_s0
        episodes_dict["QStar_S0"] = env_optimal_q_s0
    return summ_dict
