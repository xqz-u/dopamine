#!/usr/bin/env python3

import attr
import gin
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from thesis import utils as u
from thesis.jax.agents.dqv_family import dqv_base


def extract_best_action(q_values, *_, **__) -> jnp.DeviceArray:
    return jnp.max(q_values, 1)


@gin.configurable
@attr.s(auto_attribs=True)
class JaxDQVMaxAgent(dqv_base.DQV):
    Q_online: FrozenDict = FrozenDict()
    Q_target: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()

    def build_networks(self):
        self.rng, rng0, rng1 = super().build_networks()
        self.Q_online = self.Q_network.init(rng0, self.state)
        self.Q_target = self.Q_online
        self.V_online = self.V_network.init(rng1, self.state)

    def sync_weights(self):
        self.Q_target = self.Q_online

    def agent_train_step(self, replay_elements: dict):
        td_error_replay_els = u.mget(
            replay_elements, "next_state", "reward", "terminal"
        )
        v_td_targets = dqv_base.dqv_family_td_error(
            self.Q_network,
            self.Q_target,
            *td_error_replay_els,
            self.exp_data.gamma,
            extract_best_action,
        )
        q_td_targets = dqv_base.dqv_family_td_error(
            self.V_network,
            self.V_online,
            *td_error_replay_els,
            self.exp_data.gamma,
            dqv_base.mask_v_estimates,
        )
        self.Q_optim_state, self.Q_online, q_loss = dqv_base.train_module(
            self.Q_network,
            self.Q_online,
            q_td_targets,
            self.optimizer,
            self.Q_optim_state,
            self.exp_data.loss_fn,
            replay_elements["state"],
            dqv_base.mask_q_estimates,
            replay_elements["action"],
        )
        self.V_optim_state, self.V_online, v_loss = dqv_base.train_module(
            self.V_network,
            self.V_online,
            v_td_targets,
            self.optimizer,
            self.V_optim_state,
            self.exp_data.loss_fn,
            replay_elements["state"],
            dqv_base.mask_v_estimates,
        )
        return q_loss, v_loss

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        return super().bundle_and_checkpoint(
            checkpoint_dir,
            iteration_number,
            V_online=self.V_online,
            Q_online=self.Q_online,
            Q_target=self.Q_target,
        )
