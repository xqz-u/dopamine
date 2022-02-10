from typing import Tuple

import attr
from jax import numpy as jnp
from thesis.runner import runner_base


@attr.s(auto_attribs=True)
class OnlineRunner(runner_base.Runner):
    def run_one_episode(self, mode: str) -> Tuple[int, float, jnp.DeviceArray]:
        loss = self.agent.init_loss()
        episode_steps, episode_reward, done = 0, 0.0, False
        observation = self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(action, episode_steps)
            self.agent.record_trajectory(reward, done)
            loss += self.agent.learn()
            episode_reward += reward
            episode_steps += 1
        return episode_steps, episode_reward, loss

    def run_episodes(self, mode: str) -> dict:
        n_episodes, tot_steps, tot_reward = 0, 0, 0.0
        tot_loss, loss_steps = self.agent.init_loss(), 0
        while tot_steps < self.steps:
            (
                episode_steps,
                episode_reward,
                episode_losses,
            ) = self.run_one_episode(mode)
            if self.agent.trainable:
                loss_steps += tot_steps
            tot_reward += episode_reward
            tot_steps += episode_steps
            n_episodes += 1
            tot_loss += episode_losses
        return {
            "return": tot_reward,
            "episodes": n_episodes,
            "steps": tot_steps,
            "loss_steps": loss_steps,
            "losses": tot_loss,
        }

    @property
    def console_name(self):
        return __name__
