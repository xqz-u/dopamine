import attr
import numpy as np
from thesis.agents import agent_base


# TODO agnostic on networks names? and then set them as derived class
# @property es? or just do the latter without the first?
@attr.s(auto_attribs=True)
class DQVMaxAgent(agent_base.Agent):
    def build_networks_and_optimizers(self):
        net_names, out_dims = ["qnet", "vnet"], [self.num_actions, 1]
        self._build_networks_and_optimizers(net_names, out_dims)
        # initialize target q network weights with online ones
        qnet_params = self.models["qnet"].params
        self.models["qnet"].params = {
            "online": qnet_params,
            "target": qnet_params,
        }

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(
            obs,
            self.models["qnet"].net,
            self.models["qnet"].params["online"],
        )

    def agent_train_step(self):
        pass

    def sync_weights(self):
        self.models["qnet"].params["target"] = self.models["qnet"].params["online"]

    # TODO modify unbundle and what else the runner uses to allow for
    # nested dictionaries, also because I do not think the unbundle
    # agent_base method can deal with them correctly rn
    def bundle_and_checkpoint(self, checkpoint_dir: str, iteration_number: int) -> dict:
        q_params = self.models["qnet"].params
        q_dict = {
            "q_params_online": q_params["online"],
            "q_params_target": q_params["target"],
            "q_opt_state": self.models["qnet"].opt_state,
        }
        v_dict = {
            "v_params": self.models["vnet"].params,
            "v_opt_state": self.models["vnet"].opt_state,
        }
        return {
            **q_dict,
            **v_dict,
            **super().bundle_and_checkpoint(checkpoint_dir, iteration_number),
        }

    def unbundle(self):
        # do something here
        pass


import optax
from thesis import exploration, networks

conf = {
    "nets": {
        "qnet": {
            "model": {
                "class_": networks.mlp,
                "hiddens": (5, 5),
            },
            "optim": {
                "class_": optax.sgd,
                "learning_rate": 0.09,
            },
        },
        "vnet": {"optim": {"learning_rate": 0.05}},
    },
    "exploration": {
        "fn": exploration.egreedy_linear_decay,
        # "fn": exploration.egreedy,
    },
    "memory": {
        "stack_size": 1,
        "replay_capacity": 50000,
        "batch_size": 128,
    },
    "experiment": {
        "seed": 4,
        "steps": 1000,
        "iterations": 500,
        "redundancy": 5,
        "env": "CartPole-v0",
    },
    "agent": {
        "class_": DQVMaxAgent,
        "net_sync_freq": 2000,
        "min_replay_history": 5000,
        "train_freq": 4,
    },
    # "logs": {},
}


# def make_agent(config: dict):
#     return config["agent"]["class_"](
#         config, num_actions=2, observation_shape=(4, 1), observation_dtype=np.float64
#     )


# max_ag = make_agent(conf)
