from typing import Dict, Tuple

import attr
import numpy as np
from jax import numpy as jnp
from thesis.agents import Agent, DQVMaxAgent, agent_utils


@attr.s(auto_attribs=True)
class DQNAgent(Agent.Agent):
    @property
    def model_names(self) -> Tuple[str]:
        return ("qnet",)

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [self.num_actions])
        qnet_params = self.models["qnet"].params
        self.models["qnet"].params = {"online": qnet_params, "target": qnet_params}

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return DQVMaxAgent.DQVMaxAgent.select_action(self, obs)

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Tuple[jnp.DeviceArray]:
        td_error = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["qnet"].net,
                self.models["qnet"].params["target"],
                replay_elts["next_state"],
            ).max(1),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        (
            self.models["qnet"],
            self.models["qnet"].params["online"],
            q_loss,
        ) = DQVMaxAgent.train_q_net(
            self.gamma,
            self.models["qnet"],
            self.models["qnet"].params["online"],
            replay_elts["state"],
            replay_elts["action"],
            td_error,
        )
        return (q_loss,)

    def sync_weights(self):
        return DQVMaxAgent.DQVMaxAgent.sync_weights(self)
