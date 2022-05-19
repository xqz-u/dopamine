import gym
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrand


def print_model(model_name: str, params):
    print(f"{model_name}: {params}\nstructure: {jax.tree_map(jnp.shape, params)}")


# compute the temporal difference target, given value estimates for
# the states at t+1
def td_error(
    rewards: jnp.ndarray,
    state_t1_estim: jnp.ndarray,
    terminal_t1: jnp.ndarray,
    gamma: float = 0.99,
) -> jnp.ndarray:
    return rewards + gamma * state_t1_estim * (1.0 - terminal_t1)


# evalueate batched inputs on a linen module (flax's NNs)
batch_net_eval = lambda model, params, inputs: jax.vmap(
    lambda x: model.apply(params, x)
)(inputs).squeeze()


# mean squared loss
mse_loss = lambda targets, predictions: jnp.power(targets - predictions, 2)


env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape

rng = jrand.PRNGKey(0)

# create 2 simple perceptrons as Q and V functions
qfunc = nn.Dense(features=env.action_space.n)
vfunc = nn.Dense(features=1)
rng, k, sk = jrand.split(rng, 3)
q_params = qfunc.init(k, jnp.zeros(obs_shape))
v_params = vfunc.init(sk, jnp.zeros(obs_shape))
print_model("Q-Func", q_params)
print_model("V-Func", v_params)

minibatch_size = 8
print(f"minibatch_size: {minibatch_size}")

# simulate some CartPole data
states_t0 = np.random.uniform(size=(minibatch_size,) + obs_shape)
states_t1 = np.random.uniform(size=(minibatch_size,) + obs_shape)
actions_t0 = np.random.randint(0, env.action_space.n, size=(minibatch_size,))
# reward for CartPole is +1 for each time step the pole is up
terminals_t1 = np.random.randint(0, env.action_space.n, size=(minibatch_size,))
rewards_t0 = 1.0 - terminals_t1

# simulated temporal difference targets, as in line 19 of the DQV
# pseudocode in the paper
v_states_t1 = batch_net_eval(vfunc, v_params, states_t1)
td_targets = td_error(rewards_t0, v_states_t1, terminals_t1)


# Q-values for every action of the states sampled for replay
qs = batch_net_eval(qfunc, q_params, states_t0)
# actually consider only the action-values of the actions that were
# taken in those states; together with line above, this is the
# subtrahend on line 24 in DQV paper pseudocode
chosen_qs = jax.vmap(lambda x, y: x[y])(qs, actions_t0)
print(f"Q-values of replayed actions:\n{chosen_qs}")

# loss on line 24 in DQV paper pseudocode
loss = jnp.mean(jax.vmap(mse_loss)(td_targets, chosen_qs))
print(f"MSE (targets are simulated): {loss}")


# NOTE question
# I do not get this: why would I get the max on chosen_qs?
# they are multiple ones because we are evaluating the Q function on a
# batched input; but each value is an estimate in itself. Isn't then
# taking the maximum just picking out a candidate from a population,
# with no real difference then taking the mean of this values - in the
# long run, both the max and the mean will increase because of the
# distributional shift?
max_chosen_q = chosen_qs.max()
print(f"max reported Q value: {max_chosen_q}")
