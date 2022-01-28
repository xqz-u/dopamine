import logging
import os
import pprint
from copy import deepcopy
from typing import Dict, List, Tuple

import attr
import jax
import numpy as np
from dopamine.discrete_domains import gym_lib
from jax import numpy as jnp
from thesis import constants, custom_pytrees, patcher, utils
from thesis.runner import reporter


def create_runner(conf: dict):
    conf["runner"]["schedule"] = conf["runner"].get("schedule", "train")
    schedule = conf["runner"]["schedule"]
    if schedule == "train":
        runner = TrainRunner
    elif schedule == "continuous_train_and_eval":
        runner = Runner
    else:
        raise ValueError(f"Unknown runner schedule: {schedule}")
    return runner(conf, **conf["runner"]["experiment"])


@attr.s(auto_attribs=True)
class Runner:
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
        ckpt_dir = os.path.join(self.conf["runner"]["base_dir"], "checkpoints")
        self._checkpointer = patcher.Checkpointer(ckpt_dir)
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

    # TODO take restarting into account
    # TODO move self.console out of class? should still initilize name
    # and level
    def setup_reporters(self):
        self.console = utils.ConsoleLogger(
            level=self.conf["runner"].get("log_level", logging.DEBUG), name=__name__
        )
        # for rep in self.conf["runner"].get("reporters"):
        #     reporter_ = rep["call_"]
        #     self.reporters.append(reporter_(**utils.argfinder(reporter_, rep)))

    def try_resuming(self) -> bool:
        # Check if checkpoint exists. Note that the existence of
        # checkpoint 0 means that we have finished iteration 0 (so we
        # will start from iteration 1).
        latest_ckpts = patcher.get_latest_ckpt_number(
            self._checkpointer._base_directory
        )
        if latest_ckpts == -1:
            return False
        redundancy, iteration = latest_ckpts
        agent_data = self._checkpointer._load_checkpoint(redundancy, iteration)
        if agent_data is None:
            self.console.warning("Unable to reload the agent's parameters!")
            return False
        # FIXME not optimal; would be better to:
        # - handle this agent.unbundle, so might require agent's not to
        #   bulid networks
        # - or, when bundling, save the full rng: there could have been
        #   millions of splits
        # restore agent with previous rng
        # NOTE this overwrites seed if, in the config, it changed
        self.seed = agent_data["seed"]
        self.create_agent(agent_data["n_splits"])
        self.agent.unbundle(
            self._checkpointer._base_directory, redundancy, iteration, agent_data
        )

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

    def report_metrics(
        self, run_mode: str, reports: dict, step=None, epoch=None, **kwargs
    ) -> List[Dict[str, List[Tuple[str, float]]]]:
        return [
            reporter_(
                reports,
                step=step,
                epoch=epoch,
                context={"subset": run_mode, **kwargs},
            )
            for reporter_ in self.reporters
        ]

    def run_one_episode(
        self, mode: str, losses: jnp.DeviceArray
    ) -> Tuple[int, float, jnp.DeviceArray]:
        episode_steps, episode_reward, done = 0, 0.0, False
        observation = self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.env.step(action)
            done = done or episode_steps >= self.env.environment.spec.max_episode_steps
            if self.conf["env"]["clip_rewards"]:
                reward = np.clip(reward, -1, 1)
            episode_reward += reward
            episode_steps += 1
            step_loss = self.agent.learn(observation, reward, done)
            if step_loss is not None:
                losses += step_loss
        return episode_steps, episode_reward, losses

    def run_episodes(self, mode: str):
        loss_init = lambda: jnp.zeros((len(self.conf["nets"]), 1))
        n_episodes, tot_steps, tot_reward = 0, 0, 0.0
        tot_loss, loss_steps = loss_init(), None
        while tot_steps < self.steps:
            (
                episode_steps,
                episode_reward,
                episode_losses,
            ) = self.run_one_episode(mode, loss_init())
            if self.agent.training_steps >= self.agent.min_replay_history:
                loss_steps = tot_steps
            tot_reward += episode_reward
            tot_steps += episode_steps
            n_episodes += 1
            tot_loss += episode_losses
        self.global_steps += tot_steps
        # NOTE if not enough steps are performed loss_tags is None and
        # loss_vals 0's
        metrics = {
            "return": tot_reward,
            "episodes": n_episodes,
            "steps": tot_steps,
            "loss_steps": tot_steps - (loss_steps or 0),
            "losses": {k: float(v) for k, v in zip(self.agent.losses_names, tot_loss)},
        }
        reported = self.report_metrics(
            mode, metrics, step=self.global_steps, epoch=self.curr_iteration
        )
        self.console.debug(
            f"#{self.curr_iteration} #ep {n_episodes} #steps {tot_steps} #global {self.global_steps}\n{pprint.pformat(reported)}"
        )

    def run_one_iteration(self):
        self.agent.eval_mode = False
        self.run_episodes("train")
        self.agent.eval_mode = True
        self.console.debug("Eval round...")
        self.run_episodes("eval")

    def run_experiment(self):
        env_seed = self.next_seeds()
        self.console.debug(f"Env seeds: {env_seed} Agent rng: {self.agent.rng}")
        for reporter_ in self.reporters:
            reporter_.setup(self.hparams)
        self.console.info(pprint.pformat(self.hparams))
        while self.curr_iteration < self.iterations:
            self._checkpoint_experiment()
            self.curr_iteration += 1

    def run_experiment_with_redundancy(self):
        while self.curr_redundancy < self.redundancy:
            self.console.debug(
                f"Start: redundancy {self.curr_redundancy} iteration: {self.curr_iteration}"
            )
            self.run_experiment()
            self.curr_redundancy += 1
            self.curr_iteration, self.global_steps = 0, 0
            if self.curr_redundancy != self.redundancy:
                self.create_agent()

    def _checkpoint_experiment(self):
        agent_data = self.agent.bundle_and_checkpoint(
            self._checkpointer._base_directory,
            self.curr_redundancy,
            self.curr_iteration,
        )
        if not agent_data:
            return
        for key in ["curr_redundancy", "curr_iteration", "global_steps"]:
            agent_data[key] = getattr(self, key)
        self._checkpointer._save_checkpoint(
            self.curr_redundancy, self.curr_iteration, agent_data
        )


@attr.s(auto_attribs=True)
class TrainRunner(Runner):
    def run_one_iteration(self):
        self.agent.eval_mode = False
        self.run_episodes("train")


# def _log_experiment(self, statistics: dict):
#     redundancy_key = f"redundancy_{self.curr_redundancy}"
#     self._logger[redundancy_key] = self._logger.get(redundancy_key, {})
#     self._logger[redundancy_key][f"iteration_{self.curr_iteration}"] = statistics
#     self._logger.log_to_file(self.curr_redundancy, self.curr_iteration)
