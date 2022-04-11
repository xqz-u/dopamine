import gym
import optax
from dopamine.jax import losses
from thesis import config, constants, custom_pytrees
from thesis.agents import DQNAgent
from thesis.memory import offline_memory

cp = gym.make("CartPole-v1")

conf = {
    "nets": {
        "qfunc": {
            "model": {"hiddens": (512, 512)},
            "optim": {"call_": optax.adam, "learning_rate": 0.001, "eps": 3.125e-4},
            "loss_metric": losses.huber_loss,
        }
    },
    "memory": {
        # "call_": offline_memory.OfflineOutOfGraphReplayBuffer,
        # "_buffers_dir": f"{config.data_dir}/CartPole-v1/DQNAgent/cp_dqn_full_experience_%%/checkpoints/redundancy_2",
        # "batch_size": 256,
        # "update_horizon": 3,
    },
    "env": {"environment_name": "CartPole", "version": "v1"},
    "exploration": {},
    "agent": {},
}

rng = custom_pytrees.PRNGKeyWrap()

agent = DQNAgent.DQNAgent(
    conf=conf, num_actions=cp.action_space.n, **constants.env_info(cp), rng=rng
)
