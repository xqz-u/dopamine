from typing import Dict, Tuple

import attr
import numpy as np
from jax import numpy as jnp
from thesis.agents import Agent, DQVMaxAgent, agent_utils


@attr.s(auto_attribs=True)
class DQVAgent(Agent.Agent):
    @property
    def model_names(self) -> Tuple[str]:
        return ("vfunc", "qfunc")

    @property
    def repr_name(self) -> str:
        return "DQV"

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, ...]:
        return self._select_action(
            obs, self.models["qfunc"].net, self.models["qfunc"].params
        )

    def build_networks_and_optimizers(self):
        self._build_networks_and_optimizers(self.model_names, [1, self.num_actions])
        vfunc_params = self.models["vfunc"].params
        self.models["vfunc"].params = {"online": vfunc_params, "target": vfunc_params}

    def train(self, replay_elts: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
        td_targets = agent_utils.td_error(
            self.gamma,
            agent_utils.batch_net_eval(
                self.models["vfunc"].net,
                self.models["vfunc"].params["target"],
                replay_elts["next_state"],
            ),
            replay_elts["reward"],
            replay_elts["terminal"],
        )
        (
            self.models["vfunc"],
            self.models["vfunc"].params["online"],
            v_loss,
        ) = DQVMaxAgent.train_v_net(
            self.models["vfunc"],
            self.models["vfunc"].params["online"],
            replay_elts["state"],
            td_targets,
        )
        (
            self.models["qfunc"],
            self.models["qfunc"].params,
            q_loss,
        ) = DQVMaxAgent.train_q_net(
            self.models["qfunc"],
            self.models["qfunc"].params,
            replay_elts["state"],
            replay_elts["action"],
            td_targets,
        )
        return {"loss": (v_loss, q_loss)}

    def sync_weights(self):
        self.models["vfunc"].params["target"] = self.models["vfunc"].params["online"]
