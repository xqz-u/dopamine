from typing import Dict, Tuple

import attr
import numpy as np
from jax import numpy as jnp
from thesis.agents import Agent, DQVMaxAgent, agent_utils


@attr.s(auto_attribs=True)
class DQVAgent(Agent.Agent):
    @property
    def model_names(self) -> Tuple[str]:
        return ("vnet", "qnet")

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(
            obs, self.models["qnet"].net, self.models["qnet"].params
        )

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [1, self.num_actions])
        vnet_params = self.models["vnet"].params
        self.models["vnet"].params = {"online": vnet_params, "target": vnet_params}

    def sync_weights(self):
        self.models["vnet"].params["target"] = self.models["vnet"].params["online"]

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Tuple[jnp.DeviceArray]:
        td_targets = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["vnet"].net,
                self.models["vnet"].params["target"],
                replay_elts["next_state"],
            ),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        (
            self.models["vnet"],
            self.models["vnet"].params["online"],
            v_loss,
        ) = DQVMaxAgent.train_v_net(
            self.gamma,
            self.models["vnet"],
            self.models["vnet"].params["online"],
            replay_elts["state"],
            td_targets,
        )
        (
            self.models["qnet"],
            self.models["qnet"].params,
            q_loss,
        ) = DQVMaxAgent.train_q_net(
            self.gamma,
            self.models["qnet"],
            self.models["qnet"].params,
            replay_elts["state"],
            replay_elts["action"],
            td_targets,
        )
        return v_loss, q_loss
