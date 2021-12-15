#!/usr/bin/env python3

import attr
import gin
import jax
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from thesis.jax.agents.dqv_family import dqv_base


@gin.configurable
@attr.s(auto_attribs=True)
class JaxDQVAgent(dqv_base.DQV):
    V_online: FrozenDict = FrozenDict()
    V_target: FrozenDict = FrozenDict()
    Q_online: FrozenDict = FrozenDict()

    def build_networks(self):
        self.rng, rng0, rng1 = super().build_networks()
        self.V_online = self.V_network.init(rng0, self.state)
        self.V_target = self.V_online
        self.Q_online = self.Q_network.init(rng1, self.state)

    def sync_weights(self):
        self.V_target = self.V_online

    def agent_train_step(self, replay_elements: dict):
        td_error = dqv_base.dqv_family_td_error(
            self.V_network,
            self.V_target,
            replay_elements["next_state"],
            replay_elements["reward"],
            replay_elements["terminal"],
            self.exp_data.gamma,
            dqv_base.mask_v_estimates,
        )
        self.V_optim_state, self.V_online, v_loss = dqv_base.train_module(
            self.V_network,
            self.V_online,
            td_error,
            self.optimizer,
            self.V_optim_state,
            self.exp_data.loss_fn,
            replay_elements["state"],
            dqv_base.mask_v_estimates,
        )
        self.Q_optim_state, self.Q_online, q_loss = dqv_base.train_module(
            self.Q_network,
            self.Q_online,
            td_error,
            self.optimizer,
            self.Q_optim_state,
            self.exp_data.loss_fn,
            replay_elements["state"],
            dqv_base.mask_q_estimates,
            replay_elements["action"],
        )
        return q_loss, v_loss

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        return super().bundle_and_checkpoint(
            checkpoint_dir,
            iteration_number,
            Q_online=self.Q_online,
            V_online=self.V_online,
            V_target=self.V_target,
        )

    # NOTE kinda useless
    @property
    def networks_shape(self):
        return dict(
            map(
                lambda attr: (
                    attr,
                    jax.tree_map(jnp.shape, getattr(self, attr)).unfreeze(),
                ),
                ["Q_online", "V_online", "V_target"],
            )
        )
