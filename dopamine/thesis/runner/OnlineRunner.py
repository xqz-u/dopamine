from typing import Tuple

import attr
from jax import numpy as jnp
from thesis.runner import Runner


@attr.s(auto_attribs=True)
class OnlineRunner(Runner.Runner):
    def run_one_episode(self) -> Tuple[int, int, float, jnp.DeviceArray]:
        loss = self.agent.init_loss()
        episode_steps, loss_steps, episode_reward, done = 0, 0, 0.0, False
        observation = self.env.reset()
        while not done:
            action = self.agent.select_action(observation)
            observation, reward, done, _ = self.step_environment(action, episode_steps)
            self.agent.record_trajectory(reward, done)
            step_loss, training_started = self.agent.learn()
            loss += step_loss
            loss_steps += training_started
            episode_reward += reward
            episode_steps += 1
        return episode_steps, loss_steps, episode_reward, loss

    def run_episodes(self) -> dict:
        n_episodes, tot_steps, tot_loss_steps, tot_reward = 0, 0, 0, 0.0
        tot_loss = self.agent.init_loss()
        while tot_steps < self.steps:
            (
                episode_steps,
                episode_loss_steps,
                episode_reward,
                episode_loss,
            ) = self.run_one_episode()
            tot_reward += episode_reward
            tot_steps += episode_steps
            tot_loss_steps += episode_loss_steps
            tot_loss += episode_loss
            n_episodes += 1
        return {
            "return": tot_reward,
            "episodes": n_episodes,
            "steps": tot_steps,
            "loss_steps": tot_loss_steps,
            "losses": tot_loss,
        }

    @property
    def console_name(self):
        return __name__
