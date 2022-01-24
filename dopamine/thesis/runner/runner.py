import logging
from copy import deepcopy
from typing import List, Tuple

import attr
import jax
import numpy as np
from dopamine.discrete_domains import (
    checkpointer,
    gym_lib,
    iteration_statistics,
    logger,
)
from jax import numpy as jnp
from thesis import custom_pytrees, utils
from thesis.runner import reporter

# NOTE if you want to have start/stop/resume functionality when running
# an experiment with redundancy, to restart a run correctly do:
# -> iterations = og iterations - max checkpointed experiment iteration
# -> redundancy = og redundancy - |redundancies already done|
# -> seed = 0 + |redundancies already done|


# TODO differentiate on schedule
def create_runner(conf: dict):
    return Runner(conf, **conf["runner"]["experiment"])


# TODO check if logger and iteration_statistics are necessary
@attr.s(auto_attribs=True)
class Runner:
    conf: dict
    seed: int = 0
    steps: int = 500
    iterations: int = 1000
    redundancy: int = 1
    checkpoint_file_prefix: str = "ckpt"
    logging_file_prefix: str = "log"
    env: object = gym_lib.create_gym_environment
    agent: object = attr.ib(init=False)
    base_dir: str = attr.ib(init=False)
    _checkpoint_dir: str = attr.ib(init=False)
    _checkpointer: checkpointer.Checkpointer = attr.ib(init=False)
    _logger: logger.Logger = attr.ib(init=False)
    start_iteration: int = 0
    curr_iteration: int = 0
    reporters: List[reporter.Reporter] = attr.ib(factory=list)

    # FIXME this way, defaults are never reported...
    @property
    def hparams(self) -> dict:
        r = deepcopy(self.conf)
        r["env"].pop("call_", None)
        for k in ["reporters", "base_dir", "resume"]:
            r["runner"].pop(k, None)
        return jax.tree_map(lambda v: f"<{v.__name__}>" if callable(v) else v, r)

    def __attrs_post_init__(self):
        self.base_dir = self.conf["runner"]["base_dir"]
        env_ = self.conf["env"].get("call_", self.env)
        self.env = env_(**utils.argfinder(env_, self.conf["env"]))
        agent_ = self.conf["agent"]["call_"]
        self.agent = agent_(
            **utils.argfinder(
                agent_,
                {
                    **self.conf["agent"],
                    **{
                        "conf": self.conf,
                        "num_actions": self.env.action_space.n,
                        "observation_dtype": self.env.observation_space.dtype,
                    },
                },
            )
        )
        self._checkpoint_dir = f"{self.base_dir}/checkpoints"
        self._logger = logger.Logger(f"{self.base_dir}/logs")
        if self.conf["runner"].get("resume", True):
            self._initialize_checkpointer_and_maybe_resume()
        # build reporters TODO better version with autobuild
        for rep in self.conf["runner"].get("reporters"):
            reporter_ = rep["call_"]
            self.reporters.append(reporter_(**utils.argfinder(reporter_, rep)))

    def next_seeds(self):
        self.env.environment.seed(self.seed)
        self.agent.rng = custom_pytrees.PRNGKeyWrap(self.seed)
        self.seed += 1

    def _initialize_checkpointer_and_maybe_resume(self):
        self._checkpointer = checkpointer.Checkpointer(
            self._checkpoint_dir, self.checkpoint_file_prefix
        )
        # Check if checkpoint exists. Note that the existence of
        # checkpoint 0 means that we have finished iteration 0 (so we
        # will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir
        )
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version
            )
            if self.agent.unbundle(
                self._checkpoint_dir, latest_checkpoint_version, experiment_data
            ):
                if experiment_data is not None:
                    assert "logs" in experiment_data
                    assert "current_iteration" in experiment_data
                    self._logger.data = experiment_data["logs"]
                    self.start_iteration = experiment_data["current_iteration"] + 1
                logging.info(
                    f"Reloaded checkpoint and will start from iteration {self.start_iteration}"
                )

    def report_metrics(
        self, run_mode: str, reports: dict, step=None, epoch=None, **kwargs
    ):
        for reporter_ in self.reporters:
            reporter_(
                reports,
                step=step,
                epoch=epoch,
                context={"subset": run_mode, **kwargs},
            )

    # NOTE no episode end for maximum episode duration (at least not
    # explicit to agent)
    def run_one_episode(
        self, mode: str, losses: jnp.DeviceArray
    ) -> Tuple[int, float, jnp.DeviceArray]:
        episode_steps, episode_reward, done = 0, 0.0, False
        observation = self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.env.step(action)
            if self.conf["env"].get("clip_rewards", False):
                reward = np.clip(reward, -1, 1)
            episode_reward += reward
            episode_steps += 1
            step_loss = self.agent.learn(observation, reward, done)
            if step_loss is not None:
                losses += step_loss
        return episode_steps, episode_reward, losses

    def run_episodes(
        self,
        steps: int,
        mode: str,
        stats: iteration_statistics.IterationStatistics,
    ):
        loss_init = lambda: jnp.zeros((len(self.conf["nets"]), 1))
        n_episodes, tot_steps, tot_reward = 0, 0, 0.0
        tot_loss, loss_steps = loss_init(), None
        while tot_steps < steps:
            (
                episode_steps,
                episode_reward,
                episode_losses,
            ) = self.run_one_episode(mode, loss_init())
            stats.append(
                {
                    f"{mode}_episode_lengths": episode_steps,
                    f"{mode}_episode_returns": episode_reward,
                }
            )
            if self.agent.training_steps >= self.agent.min_replay_history:
                loss_steps = tot_steps
            tot_reward += episode_reward
            tot_steps += episode_steps
            n_episodes += 1
            tot_loss += episode_losses
        mean_return = tot_reward / n_episodes  # if num_episodes > 0 else 0.0
        stats.append({f"{mode}_mean_return": mean_return})
        logging.info(f"Average undiscounted return per {mode} episode: {mean_return}")
        # NOTE if not enough steps are performed loss_tags is None and
        # loss_vals 0's
        metrics = {
            "return": tot_reward,
            "episodes": n_episodes,
            "steps": tot_steps,
            "loss_steps": tot_steps - (loss_steps or 0),
            "losses": {k: float(v) for k, v in zip(self.agent.losses_names, tot_loss)},
        }
        self.report_metrics(mode, metrics, step=self.curr_iteration)

    def run_one_iteration(self, steps: int) -> dict:
        stats = iteration_statistics.IterationStatistics()
        logging.info(f"Starting iteration {self.curr_iteration}")
        self.agent.eval_mode = False
        self.run_episodes(steps, "train", stats)
        # self.agent.eval_mode = True
        # self.run_episodes(steps, "eval", stats)
        return stats.data_lists

    def run_experiment(self, steps: int, iterations: int):
        # FIXME
        # if iterations <= self.start_iteration:
        #     logging.warning(
        #         f"iterations ({iterations}) < start_iteration({self.start_iteration})"
        #     )
        #     return
        self.curr_iteration = self.start_iteration
        while self.curr_iteration < iterations:
            stats = self.run_one_iteration(steps)
            # self._log_experiment(self.curr_iteration, stats)
            # self._checkpoint_experiment(self.curr_iteration)
            self.curr_iteration += 1

    def run_experiment_with_redundancy(
        self, steps: int = None, iterations: int = None, redundancy: int = None
    ):
        steps = steps or self.steps
        iterations = iterations or self.iterations
        redundancy = redundancy or self.redundancy
        for i in range(redundancy):
            logging.info(f"{i}: Beginning training...")
            self.next_seeds()
            for reporter_ in self.reporters:
                reporter_.setup(i, self.hparams)
            self.run_experiment(steps, iterations)

    def _log_experiment(self, iteration: int, statistics):
        self._logger[f"iteration_{iteration}"] = statistics
        self._logger.log_to_file(self.logging_file_prefix, iteration)

    def _checkpoint_experiment(self, iteration: int):
        experiment_data = self.agent.bundle_and_checkpoint(
            self._checkpoint_dir, iteration
        )
        if experiment_data:
            experiment_data["current_iteration"] = iteration
            experiment_data["logs"] = self._logger.data
            self._checkpointer.save_checkpoint(iteration, experiment_data)
