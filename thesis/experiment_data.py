#!/usr/bin/env python3

from dataclasses import dataclass

import gin

from dopamine.agents.dqn import dqn_agent
from dopamine.jax import losses


# different action selection from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer rs in the
# agent?
@gin.configurable
@dataclass
class ExperimentData:
    seed: int = None
    gamma: float = 0.99
    epsilon_train: float = 0.01
    epsilon_eval: float = 0.001
    update_horizon: int = 1
    stack_size: int = dqn_agent.NATURE_DQN_STACK_SIZE
    replay_capacity: int = int(1e6)
    batch_size: int = 32
    min_replay_history: int = 20000
    update_period: int = 4  # NOTE not using this rn
    target_update_period: int = 8000
    learning_rate: float = 0.01
    loss_fn: callable = losses.huber_loss
    checkpoint_dir: str = None
    checkpoint_iterations: list = None

    @property
    def train_mode(self) -> str:
        return (
            "online"
            if (self.checkpoint_dir is None and self.checkpoint_iterations is None)
            else "offline"
        )

    @property
    def online(self) -> bool:
        return self.train_mode == "online"

    def replay_buffers_view(self) -> dict:
        attrs = {
            attr: getattr(self, attr)
            for attr in ["stack_size", "batch_size", "update_horizon", "gamma"]
        }
        if self.online:
            attrs["replay_capacity"] = self.replay_capacity
        return attrs
