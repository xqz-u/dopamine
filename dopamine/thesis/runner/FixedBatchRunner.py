from collections import OrderedDict

import attr
from jax import numpy as jnp
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
        loss, q_estims = self.agent.init_loss(), jnp.array([])
        for _ in range(self.steps):
            train_dict = self.agent.learn()
            loss += train_dict["loss"]
            q_estims = jnp.concatenate([q_estims, train_dict["q_estimates"]])
            # if self.global_steps % self.eval_period == 0:
            #     self.eval_round()
        self.global_steps += self.steps
        train_info = OrderedDict(loss=loss, steps=self.steps, q_estimates=q_estims)
        aggregate_info = Runner.aggregate_losses(self.agent.loss_names, train_info)
        self.report_metrics(train_info, aggregate_info)
        return {"raw": train_info, "aggregate": aggregate_info}

    def eval_iteration(self):
        pass

    def train_and_eval_iteration(self):
        pass

    @property
    def console_name(self):
        return __name__


#     def eval_round(self):
#         self.agent.eval_mode = True
#         tot_steps, tot_return = 0, 0.0
#         for _ in range(self.eval_episodes):
#             episode_steps, episode_return = GrowingBatchRunner.collect_experiences(self)
#             tot_steps += episode_steps
#             tot_return += episode_return
#         self.do_reports(
#             {
#                 "return": tot_return,
#                 "episodes": self.eval_episodes,
#                 "steps": tot_steps,
#                 "loss_steps": 1,  # avoid division by 0
#                 "losses": self.agent.init_loss(),
#             }
#         )
#         self.agent.eval_mode = False
