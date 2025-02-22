import operator
from collections import OrderedDict

import attr
from thesis import utils
from thesis.runner import Runner


@attr.s(auto_attribs=True)
class OnlineRunner(Runner.Runner):
    record_experience: bool = False

    # NOTE when record_experience is True, start&stop does not apply
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.record_experience:
            self.agent.memory.full_experience_initializer(
                self.checkpoint_dir, self.steps, self.iterations
            )
            self.console.debug(
                f"Save full experience to {self.agent.memory._full_experience_path}"
            )

    def finalize_experiment(self):
        # record pending transitions when registering a full run's
        # experience
        if self.record_experience:
            self.agent.memory.finalize_full_experience()
            self.console.debug("Finalized experience record")
        super().finalize_experiment()

    def train_one_episode(self) -> OrderedDict:
        ret_dict = OrderedDict(reward=0.0, steps=0, loss=self.agent.init_loss())
        done, observation = False, self.env.reset()
        while not done:
            if self._render_gym:
                self.env.environment.render()
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(
                action, ret_dict["steps"]
            )
            self.agent.record_trajectory(reward, done)
            utils.inplace_dict_assoc(
                ret_dict,
                operator.add,
                update_dict={"reward": reward, "steps": 1, **self.agent.learn()},
            )
        return ret_dict

    def train_iteration(self) -> dict:
        train_info = OrderedDict(
            reward=0.0,
            steps=0,
            episodes=0,
            loss=self.agent.init_loss(),
        )
        while train_info["steps"] < self.steps:
            # this works as long as the dict returned by
            # train_one_episode has the same keys as train_info
            utils.inplace_dict_assoc(
                train_info, operator.add, update_dict=self.train_one_episode()
            )
            train_info["episodes"] += 1
        self.global_steps += train_info["steps"]
        aggregate_info = Runner.aggregate_losses(self.agent.loss_names, train_info)
        aggregate_info.update(
            {"AvgEp_return": train_info["reward"] / train_info["episodes"]}
        )
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    def _checkpoint_experiment(self):
        if self.record_experience:
            self._checkpoint_agent()
        else:
            super()._checkpoint_experiment()

    @property
    def console_name(self):
        return __name__
