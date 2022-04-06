import logging
import operator
import os
import pprint
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Tuple

import attr
import gym
import numpy as np
from dopamine.discrete_domains import checkpointer, gym_lib
from thesis import constants, custom_pytrees, utils
from thesis.agents import Agent
from thesis.reporter import reporter


# TODO set numpy random number generator and python random generator's
# seed, used by Dopamine!
# TODO to do evaluation the way the runner works right now, the agent
# models should be loaded from disk; this must be taken into account
# when creating an agent under the 'eval' schedule. for now only do
# train_and_eval indeed...
@attr.s(auto_attribs=True)
class Runner(ABC):
    conf: dict
    schedule: str = "train"
    seed: int = 0
    steps: int = 500
    iterations: int = 1000
    redundancy_nr: int = 0
    eval_steps: int = 500
    eval_period: int = 1
    env: gym.Env = gym_lib.create_gym_environment
    reporters: Dict[str, reporter.Reporter] = attr.ib(factory=dict)
    agent: Agent.Agent = attr.ib(init=False)
    curr_iteration: int = attr.ib(init=False, default=0)
    global_steps: int = attr.ib(init=False, default=0)
    console: utils.ConsoleLogger = attr.ib(init=False)
    checkpoint_dir: str = attr.ib(init=False)
    _checkpointer: checkpointer.Checkpointer = attr.ib(init=False)
    _render_gym: bool = attr.ib(init=False, default=False)

    def pprint_conf(self):
        pprint.pprint(utils.reportable_conf(self.conf))

    @property
    def current_schedule(self) -> str:
        return "eval" if self.agent.eval_mode else "train"

    def __attrs_post_init__(self):
        self.console = utils.ConsoleLogger(
            level=self.conf["runner"].get("log_level", logging.DEBUG),
            name=self.console_name,
        )
        # add values to config if it had defaults
        self.conf["runner"]["experiment"].update(
            {
                k: getattr(self, k)
                for k in [
                    "seed",
                    "steps",
                    "iterations",
                    "redundancy_nr",
                    "schedule",
                    "eval_steps",
                    "eval_period",
                ]
            },
        )
        env_ = self.conf["env"].get("call_", self.env)
        self.env = env_(**utils.argfinder(env_, self.conf["env"]))
        self.conf["env"].update(constants.env_preproc_info(**self.conf["env"]))
        self.conf["env"]["clip_rewards"] = self.conf.get("clip_rewards", False)
        self._render_gym = (
            isinstance(self.env, gym_lib.GymPreprocessing)
            and self.conf["env"].get("render_mode") == "human"
        )
        self.checkpoint_dir = os.path.join(
            self.conf["runner"]["base_dir"], "checkpoints", str(self.redundancy_nr)
        )
        self._checkpointer = checkpointer.Checkpointer(self.checkpoint_dir)
        if not self.try_resuming():
            self.create_agent()
        self.setup_reporters()

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
            **constants.env_info(self.env),
            "rng": rng,
            **utils.argfinder(agent_, {**self.conf["agent"], **self.conf["memory"]}),
        }
        self.agent = agent_(**agent_args)

    # default reporters: console and mongodb
    def setup_reporters(self):
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
                **utils.argfinder(reporter_, {**rep, **utils.attr_fields_d(self)}),
            )

    def try_resuming(self) -> bool:
        latest_iter_ckpt = checkpointer.get_latest_checkpoint_number(
            self.checkpoint_dir
        )
        if latest_iter_ckpt < 0:
            self.console.debug(
                "No previous iteration found, start experiment from scratch"
            )
            return False
        agent_data = self._checkpointer.load_checkpoint(latest_iter_ckpt)
        if agent_data is None:
            self.console.warning(f"Unable to reload checkpoint {latest_iter_ckpt}")
            return False
        # restore agent with previous rng NOTE this overwrites seed if
        # it changed in the config
        self.seed = agent_data["rng"]["seed"]
        self.create_agent(agent_data["rng"])
        self.agent.unbundle(self.checkpoint_dir, latest_iter_ckpt, agent_data)
        for key in ["curr_iteration", "global_steps"]:
            assert key in agent_data, f"{key} not in agent data."
            setattr(self, key, agent_data[key])
        self.console.info(
            f"Reloaded checkpoint: {self.redundancy_nr}-{self.curr_iteration}"
        )
        # the existence of checkpoint 0 means that we have finished
        # iteration 0 (so we will start from iteration 1)
        self.curr_iteration += 1
        return True

    def next_seeds(self):
        self.env.environment.reset(seed=self.seed)
        self.seed += 1

    def step_environment(
        self, action: int, episode_steps: int
    ) -> Tuple[np.array, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        done = done or episode_steps >= self.env.environment.spec.max_episode_steps
        if self.conf["env"]["clip_rewards"]:
            reward = np.clip(reward, -1, 1)
        return observation, reward, done, info

    def report_metrics(self, raw_metrics: dict, agg_metrics: dict):
        runner_info = {
            attrib: getattr(self, attrib)
            for attrib in [
                "curr_iteration",
                "global_steps",
                "schedule",
                "current_schedule",
            ]
        }
        for reporter_name, reporter_ in self.reporters.items():
            reporter_(raw_metrics, agg_metrics, runner_info)

    # NOTE this and the corresponding finalize method can be
    # overridden in child classes to implement specific logic that is
    # run for every loop
    def setup_experiment(self):
        self.next_seeds()
        self.console.debug(f"Env seeded, Agent rng: {self.agent.rng}")

    def finalize_experiment(self):
        # flush any buffered mongo documents
        self.reporters["mongo"].collection.safe_flush_docs()
        self.console.debug("flushed mongo reporter...")

    def run_experiment(self):
        self.setup_experiment()
        self.pprint_conf()
        while self.curr_iteration < self.iterations:
            metrics = getattr(self, f"{self.schedule}_iteration")()
            self.console.debug(
                f"{self.schedule}: #{self.curr_iteration} #global {self.global_steps}\n{pprint.pformat(metrics)}"
            )
            if not self.agent.eval_mode:
                self._checkpoint_experiment()
                self.console.debug(
                    f"wrote checkpoint {self.redundancy_nr}-{self.curr_iteration}"
                )
            self.curr_iteration += 1
        self.finalize_experiment()

    def _checkpoint_agent(self):
        if not os.path.exists(self.checkpoint_dir):
            return
        agent_data = self.agent.checkpoint_dict(
            self.checkpoint_dir, self.curr_iteration
        )
        if not agent_data:
            return
        agent_data["curr_iteration"] = self.curr_iteration
        agent_data["global_steps"] = self.global_steps
        self._checkpointer.save_checkpoint(self.curr_iteration, agent_data)
        self.console.debug("Saved agent + runner's state")

    def _checkpoint_replay_buffer(self):
        if not os.path.exists(self.checkpoint_dir):
            self.agent.memory.save(self.checkpoint_dir, self.curr_iteration)
            self.console.debug("Saved replay buffer")

    def _checkpoint_experiment(self):
        self._checkpoint_replay_buffer()
        self._checkpoint_agent()

    @abstractmethod
    def train_iteration(self) -> dict:
        pass

    @property
    @abstractmethod
    def console_name(self):
        pass

    def eval_one_episode(self) -> OrderedDict:
        ep_reward, ep_steps = 0.0, 0
        done, observation = False, self.env.reset()
        while not done:
            if self._render_gym:
                self.env.environment.render()
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(action, ep_steps)
            ep_reward += reward
            ep_steps += 1
        return OrderedDict(reward=ep_reward, steps=ep_steps)

    # NOTE default implementations provided, override if necessary
    def eval_iteration(self) -> dict:
        self.agent.eval_mode = True
        eval_info = OrderedDict(reward=0.0, steps=0, episodes=0)
        while eval_info["steps"] < self.eval_steps:
            utils.inplace_dict_assoc(
                eval_info, operator.add, update_dict=self.eval_one_episode()
            )
            eval_info["episodes"] += 1
        aggregate_info = {"AvgEp_return": eval_info["reward"] / eval_info["episodes"]}
        self.report_metrics(eval_info, aggregate_info)
        self.agent.eval_mode = False
        return {"raw": eval_info, "aggregate": aggregate_info}

    def train_and_eval_iteration(self) -> dict:
        train_dict = self.train_iteration()
        eval_dict = None
        if self.curr_iteration % self.eval_period == 0:
            eval_dict = self.eval_iteration()
        return (
            {"train": train_dict}
            if eval_dict is None
            else {"train": train_dict, "eval": eval_dict}
        )


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
