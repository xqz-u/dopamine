import logging
import os
import pprint
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple

import attr
import jax
import numpy as np
from dopamine.discrete_domains import gym_lib
from thesis import constants, custom_pytrees, patcher, utils
from thesis.runner import reporter

# NOTE to do evaluation the way the runner works right now, the agent
# models must be reloaded


@attr.s(auto_attribs=True)
class Runner(ABC):
    conf: dict
    seed: int = 0
    steps: int = 500
    iterations: int = 1000
    redundancy: int = 1
    env: object = gym_lib.create_gym_environment
    agent: object = attr.ib(init=False)
    curr_iteration: int = attr.ib(init=False, default=0)
    curr_redundancy: int = attr.ib(init=False, default=0)
    global_steps: int = attr.ib(init=False, default=0)
    reporters: List[reporter.Reporter] = attr.ib(factory=list)
    console: utils.ConsoleLogger = attr.ib(init=False)
    _checkpointer: patcher.Checkpointer = attr.ib(init=False)

    @property
    def hparams(self) -> dict:
        r = deepcopy(self.conf)
        for k in ["reporters", "base_dir", "ckpt_file_prefix", "logging_file_prefix"]:
            r["runner"].pop(k, None)
        r["env"].pop("preproc", None)
        return jax.tree_map(
            lambda v: f"<{v.__name__}>"
            if callable(v)
            else (str(v) if not utils.is_builtin(v) else v),
            r,
        )

    def __attrs_post_init__(self):
        # add values to config if it had defaults
        self.conf["runner"]["experiment"].update(
            {
                k: getattr(self, k)
                for k in ["seed", "steps", "iterations", "redundancy"]
            },
        )
        env_ = self.conf["env"].get("call_", self.env)
        self.env = env_(**utils.argfinder(env_, self.conf["env"]))
        self.conf["env"].update(constants.env_info(**self.conf["env"]))
        self.conf["env"]["clip_rewards"] = self.conf.get("clip_rewards", False)
        self.setup_reporters()
        self._checkpointer = patcher.Checkpointer(
            os.path.join(self.conf["runner"]["base_dir"], "checkpoints")
        )
        if not self.try_resuming():
            self.create_agent()

    def create_agent(self, seed_splits: int = 0):
        rng = custom_pytrees.PRNGKeyWrap(self.seed)
        for _ in range(seed_splits):
            next(rng)
        agent_ = self.conf["agent"]["call_"]
        # After a first run, conf is enriched with missing defaults
        # for hparams reporting; some keys get added e.g.
        # conf["memory"]["observation_shape"], which conflict with args
        # passed as keywords. Merging the dictionaries like this
        # eliminates duplicates, and it is safe because same keys share
        # same values.
        agent_args = {
            "conf": self.conf,
            "num_actions": self.env.action_space.n,
            "observation_shape": self.conf["env"]["observation_shape"],
            "observation_dtype": self.env.observation_space.dtype,
            "rng": rng,
            **utils.argfinder(agent_, {**self.conf["agent"], **self.conf["memory"]}),
        }
        self.agent = agent_(**agent_args)
        self.agent.eval_mode = (
            True if self.conf["runner"]["schedule"] == "eval" else False
        )

    # TODO move self.console out of class?
    def setup_reporters(self):
        self.console = utils.ConsoleLogger(
            level=self.conf["runner"].get("log_level", logging.DEBUG),
            name=self.console_name,
        )
        for rep in self.conf["runner"].get("reporters"):
            reporter_ = rep["call_"]
            self.reporters.append(reporter_(**utils.argfinder(reporter_, rep)))

    def try_resuming(self) -> bool:
        # Check if checkpoint exists. Note that the existence of
        # checkpoint 0 means that we have finished iteration 0 (so we
        # will start from iteration 1).
        latest_redund, latest_ckpt = patcher.get_latest_ckpt_number(
            self._checkpointer._base_directory
        )
        self._checkpointer.setup_redundancy(latest_redund)
        if latest_redund == -1 or latest_ckpt == -1:
            return False
        agent_data = self._checkpointer.load_checkpoint(latest_ckpt)
        if agent_data is None:
            self.console.warning("Unable to reload the agent's parameters!")
            return False
        # TODO not optimal; would be better to:
        # - when bundling, save the full rng: there could have been
        #   millions of splits
        # restore agent with previous rng
        # NOTE this overwrites seed if it changed in the config
        self.seed = agent_data["seed"]
        self.create_agent(agent_data["n_splits"])
        self.agent.unbundle(self._checkpointer._base_directory, latest_ckpt, agent_data)

        for key in ["curr_redundancy", "curr_iteration", "global_steps"]:
            assert key in agent_data, f"{key} not in agent data."
            setattr(self, key, agent_data[key])
        self.console.info(
            f"Reloaded checkpoint: {self.curr_redundancy}-{self.curr_iteration}"
        )
        self.curr_iteration += 1
        self.curr_redundancy += self.curr_iteration == self.iterations
        return True

    def next_seeds(self) -> List[int]:
        env_seed = self.env.environment.seed(self.seed)
        self.seed += 1
        return env_seed

    def step_environment(
        self, action: int, episode_steps: int
    ) -> Tuple[np.array, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        done = done or episode_steps >= self.env.environment.spec.max_episode_steps
        if self.conf["env"]["clip_rewards"]:
            reward = np.clip(reward, -1, 1)
        return observation, reward, done, info

    # NOTE maybe can be united with do_reports
    def report_metrics(
        self, reports: dict, step=None, epoch=None, **kwargs
    ) -> List[Dict[str, List[Tuple[str, float]]]]:
        return [
            reporter_(
                reports,
                step=step,
                epoch=epoch,
                context={"subset": self.conf["runner"]["schedule"], **kwargs},
            )
            for reporter_ in self.reporters
        ]

    def do_reports(self, metrics: dict):
        reported = self.report_metrics(
            metrics,
            step=self.global_steps,
            epoch=self.curr_iteration,
        )
        self.console.debug(
            f"{self.conf['runner']['schedule']}: #{self.curr_iteration} #ep {metrics['episodes']} #steps {metrics['steps']} #loss_steps {metrics['loss_steps']} #global {self.global_steps}\n{pprint.pformat(reported)}"
        )

    def _run_episodes(self):
        metrics = self.run_episodes()
        self.global_steps += metrics["steps"]
        # to avoid possible division by 0 errors when training has not
        # started yet
        metrics["loss_steps"] = metrics["loss_steps"] or 1
        metrics["losses"] = {
            k: float(v) for k, v in zip(self.agent.losses_names, metrics["losses"])
        }
        self.do_reports(metrics)

    def run_one_iteration(self):
        self._run_episodes()
        self._checkpoint_experiment()
        self.curr_iteration += 1

    # NOTE this method exists to provide a default, but should be
    # overridden in derived classes if necessary
    def run_loops(self):
        while self.curr_iteration < self.iterations:
            self.run_one_iteration()

    def run_experiment(self):
        env_seed = self.next_seeds()
        self.console.debug(f"Env seeds: {env_seed} Agent rng: {self.agent.rng}")
        for reporter_ in self.reporters:
            reporter_.setup(self.hparams, self.curr_redundancy)
        self.console.info(pprint.pformat(self.hparams))
        self.run_loops()

    def run_experiment_with_redundancy(self):
        while self.curr_redundancy < self.redundancy:
            self.console.debug(
                f"Start: redundancy {self.curr_redundancy} iteration: {self.curr_iteration}"
            )
            self.run_experiment()
            self.curr_redundancy += 1
            self._checkpointer.setup_redundancy(self.curr_redundancy)
            self.curr_iteration, self.global_steps = 0, 0
            if self.curr_redundancy != self.redundancy:
                self.create_agent()

    def _checkpoint_experiment(self):
        agent_data = self.agent.bundle_and_checkpoint(
            self._checkpointer._base_directory,
            self.curr_iteration,
        )
        if not agent_data:
            return
        for key in ["curr_redundancy", "curr_iteration", "global_steps"]:
            agent_data[key] = getattr(self, key)
        self._checkpointer.save_checkpoint(self.curr_iteration, agent_data)

    @abstractmethod
    def run_episodes(self) -> dict:
        pass

    @property
    @abstractmethod
    def console_name(self):
        pass
