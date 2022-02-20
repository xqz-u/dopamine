import copy
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf
import torch
from torch import nn, optim

from thesis import memory, networks_torch, utils


def egreedy_act(
    num_actions: int, state: np.ndarray, q_net: nn.Module, eps: float = 0.01
) -> np.ndarray:
    def greedy_action():
        # makes memory usage lower, also this operation is not
        # required in gradient computations
        with torch.no_grad():
            return torch.argmax(q_net(torch.tensor(state)))

    return np.where(
        random.random() <= eps, random.randrange(num_actions), greedy_action()
    )


def td_targets(
    net: nn.Module,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    # with torch.no_grad():
    max_q_vals = net(next_states).max(1)[0].detach()
    targets = rewards + discount * max_q_vals * (1.0 - terminals)
    return targets.type(torch.DoubleTensor)


# v = td_targets(
#     torch_agent.q_net_online, els["next_state"], els["reward"], els["terminal"], 0.99
# )


def optimize(
    loss_fn: callable,
    net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    states,
    next_states,
    actions,
    rewards,
    terminals,
    discount: float = 0.99,
) -> torch.Tensor:
    states = networks_torch.normalize_states(states)
    next_states = networks_torch.normalize_states(next_states)
    bellman_targets = td_targets(target_net, next_states, rewards, terminals, discount)
    qs_replay = (
        net(states).gather(1, actions.unsqueeze(1)).squeeze().type(torch.DoubleTensor)
    )
    loss = loss_fn(qs_replay, bellman_targets)
    # reset gradients to zero
    optimizer.zero_grad()
    # compute gradients w.r.t. network parameters, given to optimizer on init
    loss.backward()
    # update parameters in network
    optimizer.step()
    return loss.item()


@dataclass
class TorchDQN:
    num_actions: int
    q_net_online: nn.Module
    replay_memory: memory.ReplayMemory
    q_optim: optim.Optimizer = None
    loss_fn: nn.modules.loss._Loss = None
    state: np.ndarray = None
    action: int = None
    training_steps: int = 0
    train_freq: int = 4
    min_replay_hist: int = 500
    target_update_freq: int = 100
    summary_writer: tf.summary.SummaryWriter = None
    summary_writing_freq: int = 500
    _avg_loss: torch.Tensor = torch.tensor(0.0)
    q_net_target: nn.Module = None

    def __post_init__(self):
        self.q_net_target = copy.deepcopy(self.q_net_online)

    def record_trajectory(self, next_obs: np.ndarray, r: np.ndarray, done: bool):
        self.replay_memory.push(self.state, self.action, r, next_obs, int(done))

    def sample_memory(self) -> Dict[str, torch.Tensor]:
        return self.replay_memory.sample()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        self.state = obs
        self.action = egreedy_act(self.num_actions, self.state, self.q_net_online)
        return self.action

    def learn(self, obs: np.ndarray, reward: float, terminal: bool):
        self.record_trajectory(obs, reward, terminal)
        if terminal:
            return
        # if len(self.replay_memory) >= self.replay_memory.batch_size:
        if self.training_steps >= self.min_replay_hist:
            if self.training_steps % self.train_freq == 0:
                replay_elts = self.sample_memory()
                loss = optimize(
                    self.loss_fn,
                    self.q_net_online,
                    self.q_net_target,
                    self.q_optim,
                    replay_elts["state"],
                    replay_elts["next_state"],
                    replay_elts["action"],
                    replay_elts["reward"],
                    replay_elts["terminal"],
                )
                if self.can_summarise:
                    utils.add_summary_v2(
                        self.summary_writer,
                        [["scalar", "HuberLoss", loss]],
                        self.training_steps,
                    )
                self._avg_loss = (self._avg_loss + loss) / 2
        if self.training_steps % self.target_update_freq == 0:
            self.q_net_target.load_state_dict(self.q_net_online.state_dict())
        self.training_steps += 1

    @property
    def can_summarise(self):
        return (
            self.summary_writer is not None
            and self.training_steps > 0
            and self.training_steps % self.summary_writing_freq == 0
        )
