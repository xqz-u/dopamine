# from typing import Tuple

# import attr
# from jax import numpy as jnp
# from thesis.runner import Runner


# @attr.s(auto_attribs=True)
# class GrowingBatchRunner(Runner.Runner):
#     fitting_steps: int = None

#     def collect_experiences(self) -> Tuple[int, float]:
#         episode_steps, episode_reward, done = 0, 0.0, False
#         observation = self.env.reset()
#         while not done:
#             action = self.agent.select_action(observation)
#             observation, reward, done, _ = self.step_environment(action, episode_steps)
#             self.agent.record_trajectory(reward, done)
#             episode_reward += reward
#             episode_steps += 1
#         return episode_steps, episode_reward

#     def fit(self, train_steps: int) -> Tuple[jnp.DeviceArray, int]:
#         self.console.info(f"Fit {train_steps} times...")
#         loss, loss_steps = self.agent.init_loss(), 0
#         for _ in range(train_steps):
#             step_loss, training_started = self.agent.learn()
#             loss += step_loss
#             loss_steps += training_started
#         return loss, loss_steps

#     def run_episodes(self) -> dict:
#         n_episodes, tot_steps, tot_reward = 0, 0, 0.0
#         while tot_steps < self.steps:
#             (episode_steps, episode_reward) = self.collect_experiences()
#             tot_reward += episode_reward
#             tot_steps += episode_steps
#             n_episodes += 1
#         loss, loss_steps = self.fit(self.fitting_steps or tot_steps)
#         return {
#             "return": tot_reward,
#             "episodes": n_episodes,
#             "steps": tot_steps,
#             "loss_steps": loss_steps,
#             "losses": loss,
#         }

#     # TODO reintroduce random policy at the start if it's the case
#     def run_loops(self):
#         # agent_act_sel = self.agent.select_action.__func__
#         # utils.bind_instance_method(
#         #     self.agent, "select_action", agent_utils.uniform_action_selection
#         # )
#         # switched = False
#         msg = (
#             "fitting_steps is undefined -> fitting steps = exploration steps"
#             if self.fitting_steps is None
#             else f"fitting_steps: {self.fitting_steps}"
#         )
#         self.console.debug(msg)
#         while self.curr_iteration < self.iterations:
#             self.run_one_iteration()
#             # if self.agent.trainable and not switched:
#             #     utils.bind_instance_method(self.agent, "select_action", agent_act_sel)
#             #     self.console.info("Switched action selection policy back to original")
#             #     switched = True

#     @property
#     def console_name(self):
#         return __name__
