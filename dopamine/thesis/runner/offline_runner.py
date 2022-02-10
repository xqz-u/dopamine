from typing import Tuple

import attr
from jax import numpy as jnp
from thesis import utils
from thesis.agents import agent_utils
from thesis.runner import runner_base


@attr.s(auto_attribs=True)
class OfflineRunner(runner_base.Runner):
    def collect_experiences(self) -> Tuple[int, float]:
        episode_steps, episode_reward, done = 0, 0.0, False
        observation = self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(action, episode_steps)
            self.agent.record_trajectory(reward, done)
            episode_reward += reward
            episode_steps += 1
        return episode_steps, episode_reward

    def fit(self, train_steps: int) -> jnp.DeviceArray:
        loss = self.agent.init_loss()
        for _ in range(train_steps):
            loss += self.agent.learn()
        return loss

    # TODO abstract away common structure: the return dict is
    # basically the same!
    def run_episodes(self, mode: str) -> dict:
        n_episodes, tot_steps, tot_reward = 0, 0, 0.0
        loss_steps = 0
        while tot_steps < self.steps:
            (episode_steps, episode_reward) = self.collect_experiences()
            if self.agent.trainable:
                loss_steps += episode_steps
            tot_reward += episode_reward
            tot_steps += episode_steps
            n_episodes += 1
        return {
            "return": tot_reward,
            "episodes": n_episodes,
            "steps": tot_steps,
            "loss_steps": loss_steps,
            "losses": self.fit(tot_steps),
        }

    def run_loops(self):
        agent_act_sel = self.agent.select_action.__func__
        utils.bind_instance_method(
            self.agent, "select_action", agent_utils.uniform_action_selection
        )
        switched = False
        while self.curr_iteration < self.iterations:
            self.run_one_iteration_wrapped()
            if self.agent.trainable and not switched:
                utils.bind_instance_method(self.agent, "select_action", agent_act_sel)
                self.console.info("Switched action selection policy back to original")
                switched = True

    @property
    def console_name(self):
        return __name__
