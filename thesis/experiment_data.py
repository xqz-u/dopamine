#!/usr/bin/env python3

from dataclasses import dataclass

import gin

from dopamine.agents.dqn import dqn_agent


# different action selection strategies?
# eval / train mode difference?
# FIXME why is `stack_size` here alone and the other related parameters in the
# agent?
@gin.configurable
@dataclass
class ExperimentData:
    seed: int = None
    gamma: float = 0.99
    epsilon: float = 0.01
    update_horizon: int = 1
    stack_size: int = dqn_agent.NATURE_DQN_STACK_SIZE
    replay_capacity: int = int(1e6)
    batch_size: int = 32
    min_replay_history: int = 20000
    update_period: int = 4  # NOTE not using this rn
    target_update_period: int = 8000
    optimizer: str = "adam"
    loss_type: str = "huber"
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
