import operator
from collections import OrderedDict

import attr
from thesis import utils
from thesis.runner import Runner


@attr.s(auto_attribs=True)
class FixedBatchRunner(Runner.Runner):
    def __attrs_post_init__(self):
        # we are not collecting new experiences at all, so start
        # fitting immediately
        self.conf["agent"]["min_replay_history"] = 0
        super().__attrs_post_init__()
        # self.iterations = 0 NOTE why did I write this?
        assert self.agent.min_replay_history == 0

    def train_iteration(self) -> OrderedDict:
        train_info = OrderedDict(
            loss=self.agent.init_loss(), steps=self.steps, q_estimates=0.0
        )
        for _ in range(self.steps):
            utils.inplace_dict_assoc(
                train_info, operator.add, update_dict=self.agent.learn()
            )
        self.global_steps += self.steps
        aggregate_info = Runner.aggregate_losses(self.agent.loss_names, train_info)
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    def eval_one_episode(self) -> OrderedDict:
        ep_reward, ep_steps = 0.0, 0
        done, observation = False, self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(action, ep_steps)
            ep_reward += reward
            ep_steps += 1
        return OrderedDict(reward=ep_reward, steps=ep_steps)

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

    @property
    def console_name(self):
        return __name__
