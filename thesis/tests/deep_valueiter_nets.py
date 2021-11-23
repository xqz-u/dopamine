import gym
import jax
from jax import numpy as jnp
from jax import random as jrand

from thesis.jax import networks

env = gym.make("CartPole-v0")
obs_shape = env.observation_space.shape
action_shape = env.action_space.n

key, subkey = jrand.split(jrand.PRNGKey(0))

init_state = jnp.zeros(obs_shape)
rand_input = jrand.uniform(subkey, obs_shape)

model = networks.ClassicControlDQNNetwork(output_dim=action_shape, hidden_features=[10])
q_params = model.init(key, init_state)
print(jax.tree_map(jnp.shape, q_params))

print(model.apply(q_params, rand_input))


model = networks.ClassicControlDVNNetwork(hidden_features=[10])
v_params = model.init(key, init_state)
print(jax.tree_map(jnp.shape, v_params))

print(model.apply(v_params, rand_input))
