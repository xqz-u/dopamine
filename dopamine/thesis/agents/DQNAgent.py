from typing import Dict, Tuple

import attr
import numpy as np
from jax import numpy as jnp
from thesis.agents import Agent, DQVMaxAgent, agent_utils


@attr.s(auto_attribs=True)
class DQNAgent(Agent.Agent):
    @property
    def model_names(self) -> Tuple[str]:
        return ("qfunc",)

    @property
    def repr_name(self) -> str:
        return "DQN"

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [self.num_actions])
        qfunc_params = self.models["qfunc"].params
        self.models["qfunc"].params = {"online": qfunc_params, "target": qfunc_params}

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, ...]:
        return DQVMaxAgent.DQVMaxAgent.select_action(self, obs)

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        td_error = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["qfunc"].net,
                self.models["qfunc"].params["target"],
                replay_elts["next_state"],
            ).max(1),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        (
            self.models["qfunc"],
            self.models["qfunc"].params["online"],
            q_loss,
            q_estimates,
        ) = DQVMaxAgent.train_q_net(
            self.models["qfunc"],
            self.models["qfunc"].params["online"],
            replay_elts["state"],
            replay_elts["action"],
            td_error,
        )
        return {"loss": (q_loss,), "q_estimates": q_estimates}

    def sync_weights(self):
        return DQVMaxAgent.DQVMaxAgent.sync_weights(self)
