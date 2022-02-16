import math

import attr
from jax import numpy as jnp
from thesis import utils
from thesis.runner import Runner
from thesis.runner.GrowingBatchRunner import GrowingBatchRunner


def nstep_termination(self, iteration_limit: int = 1000) -> bool:
    return self.curr_iteration == iteration_limit


def winning_termination(
    self, min_return: float = -math.inf, eval_episodes_n: int = 10
) -> bool:
    loss = self.agent.init_loss()
    self.agent.eval_mode = True
    win = True
    for __ in range(eval_episodes_n):
        episode_steps, episode_return = GrowingBatchRunner.collect_experiences(self)
        self.do_reports(
            {
                "return": episode_return,
                "episodes": 1,
                "steps": episode_steps,
                "loss_steps": 0,
                "losses": loss,
            }
        )
        if episode_return < min_return:
            win = False
            break
    self.agent.eval_mode = False
    return win


# given trajectories (offline):
# - perform N (parameter) training steps, sampling from experience
# - evaluate TERMINATION CRITERION (parameter), and stop if it returns
#   true
# TODO add X eval rounds to see how the agent is performing every now
# and then; may require the "context" stuff for aim to have "eval" in
# function above, if wanting to do things right
@attr.s(auto_attribs=True)
class FixedBatchRunner(Runner.Runner):
    termination_criterion: callable = nstep_termination
    termination_args: dict = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        # we are not collecting new experiences at all, plus it does
        # not make sense to have predefined number of iterations if it
        # is not required by the termination criterion
        self.conf["agent"]["min_replay_history"] = 0
        self.iterations = 0
        # self.conf["runner"]["experiment"]["iterations"] = 0
        # enrich config with defaults
        self.conf["runner"]["experiment"][
            "termination_criterion"
        ] = self.termination_criterion
        self.termination_args.update(
            {
                k: v
                for k, v in utils.callable_defaults(self.termination_criterion).items()
                if k
                not in self.conf["runner"]["experiment"].get("termination_args", {})
            }
        )
        self.conf["runner"]["experiment"]["termination_args"] = self.termination_args
        super().__attrs_post_init__()
        assert self.agent.min_replay_history == 0
        utils.bind_instance_method(
            self, "termination_criterion", self.termination_criterion
        )

    def fit(self) -> jnp.DeviceArray:
        loss = self.agent.init_loss()
        for _ in range(self.steps):
            step_loss, _ = self.agent.learn()
            loss += step_loss
        return loss

    def run_episodes(self) -> dict:
        self.global_steps += self.steps
        return {
            "return": 0,
            "episodes": 1,
            "steps": self.steps,
            "loss_steps": self.steps,
            "losses": {
                k: float(v) for k, v in zip(self.agent.losses_names, self.fit())
            },
        }

    def run_loops(self):
        while not self.termination_criterion(**self.termination_args):
            self.do_reports(self.run_episodes())
            self._checkpoint_experiment()
            self.curr_iteration += 1

    @property
    def console_name(self):
        return __name__
