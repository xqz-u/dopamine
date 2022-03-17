import logging
import os
import pprint
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple

import attr
import jax
import numpy as np
from dopamine.discrete_domains import gym_lib
from thesis import constants, custom_pytrees, patcher, utils
from thesis.reporter import reporter

# TODO try to move experience recorder related code to online runner


# NOTE to do evaluation the way the runner works right now, the agent
# models must be reloaded; this must be taken into account when
# creating an agent under the 'eval' schedule. for now only do
# train_and_eval indeed...
@attr.s(auto_attribs=True)
class Runner(ABC):
    conf: dict
    schedule: str = "train"
    seed: int = 0
    steps: int = 500
    iterations: int = 1000
    redundancy: int = 1
    env: object = gym_lib.create_gym_environment
    agent: object = attr.ib(init=False)
    curr_iteration: int = attr.ib(init=False, default=0)
    curr_redundancy: int = attr.ib(init=False, default=0)
    global_steps: int = attr.ib(init=False, default=0)
    reporters: Dict[str, reporter.Reporter] = attr.ib(factory=dict)
    console: utils.ConsoleLogger = attr.ib(init=False)
    _checkpointer: patcher.Checkpointer = attr.ib(init=False)

    @property
    def hparams(self) -> dict:
        r = deepcopy(self.conf)
        r.pop("experiment_name", None)
        for k in ["base_dir", "ckpt_file_prefix", "logging_file_prefix"]:
            r["runner"].pop(k, None)
        r["env"].pop("preproc", None)
        r.pop("reporters", None)
        return jax.tree_map(
            lambda v: f"<{v.__name__}>"
            if callable(v)
            else (str(v) if not utils.is_builtin(v) else v),
            r,
        )

    @property
    def current_schedule(self) -> str:
        return "train" if not self.agent.eval_mode else "train_and_eval"

    def __attrs_post_init__(self):
        # add values to config if it had defaults
        self.conf["runner"]["experiment"].update(
            {
                k: getattr(self, k)
                for k in ["seed", "steps", "iterations", "redundancy", "schedule"]
            },
        )
        env_ = self.conf["env"].get("call_", self.env)
        self.env = env_(**utils.argfinder(env_, self.conf["env"]))
        self.conf["env"].update(constants.env_info(**self.conf["env"]))
        self.conf["env"]["clip_rewards"] = self.conf.get("clip_rewards", False)
        self._checkpointer = patcher.Checkpointer(
            os.path.join(self.conf["runner"]["base_dir"], "checkpoints")
        )
        self.setup_reporters()
        if not self.try_resuming():
            self.create_agent()

    def create_agent(self, serial_rng: dict = None):
        rng = (
            custom_pytrees.PRNGKeyWrap(self.seed)
            if not serial_rng
            else custom_pytrees.PRNGKeyWrap.from_dict(serial_rng)
        )
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
            "eval_mode": not self.schedule == "train",
            **utils.argfinder(agent_, {**self.conf["agent"], **self.conf["memory"]}),
        }
        self.agent = agent_(**agent_args)

    # default reporters: console and mongodb
    def setup_reporters(self):
        self.console = utils.ConsoleLogger(
            level=self.conf["runner"].get("log_level", logging.DEBUG),
            name=self.console_name,
        )
        exp_name = self.conf["experiment_name"]
        mongo_class, mongo_args = reporter.MongoReporter, {}
        reporters_confs = self.conf.get("reporters", {})
        if mongo_conf := reporters_confs.get("mongo", {}):
            mongo_class = mongo_conf["call_"]
            mongo_args = utils.argfinder(mongo_class, mongo_conf)
        self.reporters["mongo"] = mongo_class(experiment_name=exp_name, **mongo_args)
        for rep_name, rep in {
            k: v for k, v in reporters_confs.items() if k != "mongo"
        }.items():
            reporter_ = rep["call_"]
            self.reporters[rep_name] = reporter_(
                experiment_name=exp_name,
                **utils.argfinder(reporter_, rep),
            )

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
        # restore agent with previous rng
        # NOTE this overwrites seed if it changed in the config
        self.seed = agent_data["rng"]["seed"]
        self.create_agent(agent_data["rng"])
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

    # {**runner_info, "collection_tag": "experiment_progression"}
    # if reporter_name == "mongo"
    # else runner_info,
    def report_metrics(self, raw_metrics: dict, agg_metrics: dict):
        runner_info = {
            attrib: getattr(self, attrib)
            for attrib in [
                "curr_redundancy",
                "curr_iteration",
                "global_steps",
                "schedule",
                "current_schedule",
            ]
        }
        for reporter_name, reporter_ in self.reporters.items():
            reporter_(raw_metrics, agg_metrics, runner_info)

    def setup_experiment(self):
        # if self.conf["runner"].get("exp_recorder"):
        #     self.agent.memory.full_experience_initializer(
        #         self._checkpointer._base_directory, self.steps, self.iterations
        #     )
        env_seed = self.next_seeds()
        self.console.debug(f"Env seeds: {env_seed} Agent rng: {self.agent.rng}")
        for reporter_ in self.reporters.values():
            reporter_.setup(self.hparams, self.curr_redundancy)

    def finalize_experiment(self):
        # flush buffered mongo documents and record pending
        # transitions when registering a full run's experience
        # if self.conf["runner"].get("exp_recorder"):
        #     self.agent.memory.finalize_full_experience()
        self.reporters["mongo"].collection.safe_flush_docs()

    def run_experiment(self):
        self.console.info(pprint.pformat(self.hparams))
        while self.curr_iteration < self.iterations:
            metrics = getattr(self, f"{self.schedule}_iteration")()
            self.console.debug(
                f"{self.current_schedule}: #{self.curr_iteration} #global {self.global_steps}\n{pprint.pformat(metrics)}"
            )
            self._checkpoint_experiment()
            self.curr_iteration += 1

    def run_experiment_with_redundancy(self):
        while self.curr_redundancy < self.redundancy:
            self.console.debug(
                f"Start: redundancy {self.curr_redundancy} iteration: {self.curr_iteration}"
            )
            self.setup_experiment()
            self.run_experiment()
            self.finalize_experiment()
            self.curr_redundancy += 1
            if self.curr_redundancy != self.redundancy:
                self._checkpointer.setup_redundancy(self.curr_redundancy)
                self.create_agent()
            self.curr_iteration, self.global_steps = 0, 0

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
    def train_iteration(self):
        pass

    @abstractmethod
    def eval_iteration(self):
        pass

    @abstractmethod
    def train_and_eval_iteration(self):
        pass

    @property
    @abstractmethod
    def console_name(self):
        pass


# NOTE that e.g. in an iteration of 600 steps, enough experience to
# start training might occur at the step 450, so dividing the total
# loss by the number of steps (and not by the number of steps since
# traning has started) is a bit optimistic only in the first tranining
# round
# NOTE when training has not started, loss and q_estimates are not
# computed; using 0 as a proxy right now, maybe it is better not to
# track at all until available
def aggregate_losses(loss_names: Tuple[str], raw_metrics: OrderedDict) -> OrderedDict:
    # cast to float to report with mongo without need to serialize
    raw_metrics["loss"] = {k: float(v) for k, v in zip(loss_names, raw_metrics["loss"])}
    raw_metrics["q_estimates"] = float(raw_metrics["q_estimates"])
    return OrderedDict(
        **{
            f"AvgStep_{k}": (v / raw_metrics["steps"] if v != 0 else v)
            for k, v in raw_metrics["loss"].items()
        },
        AvgStep_q_estimates=qs / raw_metrics["steps"]
        if (qs := raw_metrics["q_estimates"]) != 0
        else qs,
    )
