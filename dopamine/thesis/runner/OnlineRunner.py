import operator
from collections import OrderedDict

import attr
from jax import numpy as jnp
from thesis import utils
from thesis.runner import Runner

# TODO return a dict from the agent containing all the relevant info,
# such as q-values


@attr.s(auto_attribs=True)
class OnlineRunner(Runner.Runner):
    def train_one_episode(self) -> OrderedDict:
        ret_dict = OrderedDict(
            reward=0.0, steps=0, loss=self.agent.init_loss(), q_estimates=jnp.array([])
        )
        done, observation = False, self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(
                action, ret_dict["steps"]
            )
            self.agent.record_trajectory(reward, done)
            train_dict = self.agent.learn()
            utils.inplace_dict_assoc(
                ret_dict, operator.add, *[reward, 1, train_dict["loss"]]
            )
            ret_dict["q_estimates"] = jnp.concatenate(
                [ret_dict["q_estimates"], train_dict["q_estimates"]]
            )
        return ret_dict

    def train_iteration(self) -> dict:
        train_info = OrderedDict(
            reward=0.0,
            steps=0,
            episodes=0,
            loss=self.agent.init_loss(),
            q_estimates=jnp.array([]),
        )
        while train_info["steps"] < self.steps:
            episode_train_dict = self.train_one_episode()
            q_estimates = episode_train_dict.pop("q_estimates")
            # this works as long as the dict returned by
            # train_one_episode has the same keys as train_info
            utils.inplace_dict_assoc(
                train_info, operator.add, update_dict=episode_train_dict
            )
            train_info["q_estimates"] = jnp.concatenate(
                [train_info["q_estimates"], q_estimates]
            )
            train_info["episodes"] += 1
        self.global_steps += train_info["steps"]
        aggregate_info = Runner.aggregate_losses(self.agent.loss_names, train_info)
        aggregate_info.update(
            {"AvgEp_return": train_info["reward"] / train_info["episodes"]}
        )
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    def eval_iteration(self):
        pass

    def train_and_eval_iteration(self):
        pass

    @property
    def console_name(self):
        return __name__
