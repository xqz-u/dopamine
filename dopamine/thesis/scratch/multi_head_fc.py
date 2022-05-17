from typing import List, Tuple, Union

import gym
import jax
import numpy as np
import optax
from dopamine.jax import networks as jax_networks
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random as jrand
from thesis import constants, custom_pytrees, networks, utils

rng = custom_pytrees.PRNGKeyWrap()

env = gym.make("CartPole-v1")
env_spec = constants.env_info(env)

rand_input = lambda shape=env_spec["observation_shape"]: jrand.uniform(next(rng), shape)

example_input, test_input = [rand_input()] * 2


mlp = networks.mlp(output_dim=2, hiddens=(8,))
mlp_params = mlp.init(next(rng), example_input)
utils.jax_container_shapes(mlp_params)


def mse(params, x_batched, y_batched, model):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


mse_grad_fn = jax.value_and_grad(mse)

target = rand_input(shape=(4, 2))


class MultiHeadDense:
    heads: List[Tuple[nn.Dense, FrozenDict]]

    def __init__(
        self,
        features: int,
        n_heads: int,
        rng: custom_pytrees.PRNGKeyWrap,
        input_shape: tuple,
    ):
        self.heads = []
        for _ in range(n_heads):
            net = nn.Dense(features=features)
            params = net.init(next(rng), jnp.ones(input_shape))
            self.heads.append((net, params))

    def __call__(self, xs: Union[jnp.DeviceArray, np.ndarray]) -> jnp.DeviceArray:
        return jnp.asarray([head.apply(params, xs) for head, params in self.heads])


x = MultiHeadDense(2, 5, rng, env_spec["observation_shape"])
pred = x(example_input)

loss_and_grads = [
    mse_grad_fn(params, example_input, target, net) for net, params in x.heads
]


# TODO find interface thas can be written agnostically, passing all the
# requested parameters - e.g. loss_metric is missing rn, activation_fn
# NOTE would be cool to be able to pass this class to jitted functions!
# for now just use a for loop, since you can still pass
# NetworkOptimWraps...
class EnsembledNet:
    net_with_heads: List[custom_pytrees.NetworkOptimWrap]

    def __init__(
        self,
        features: int,
        n_heads: int,
        input_shape: tuple,
        shared_body: nn.Module,
        rng: custom_pytrees.PRNGKeyWrap,
    ):
        self.net_with_heads = []

        body_out_shape = (input_shape[0],)
        # TODO take last dense layer out of jax_networks.NatureDQNNetwork?
        if isinstance(shared_body, jax_networks.NatureDQNNetwork):
            body_out_shape += (...,)
        elif isinstance(shared_body, networks.Sequential) and isinstance(
            shared_body.layers[-1], (nn.Dense, networks.DensePreproc)
        ):
            body_out_shape += (shared_body.layers[-1].features,)
        else:
            raise Exception

        shared_params = shared_body.init(next(rng), jnp.zeros(input_shape))

        for i in range(n_heads):
            head = nn.Dense(features=features)
            head_params = head.init(next(rng), jnp.zeros(body_out_shape))
            shared_params = shared_params.unfreeze()
            shared_params["params"]["final_head"] = head_params["params"]
            shared_params = FrozenDict(shared_params)
            opt = optax.adam(learning_rate=0.001, eps=3.125e-4)
            opt_state = opt.init(shared_params)
            self.net_with_heads.append(
                custom_pytrees.NetworkOptimWrap(
                    shared_params,
                    opt_state,
                    networks.Sequential([shared_body, nn.relu, head]),
                    opt,
                )
            )


ensemble_mlp = EnsembledNet(env.action_space.n, 3, example_input.shape, mlp, rng)


preds = [
    custom_net.net.apply(custom_net.params, test_input)
    for custom_net in ensemble_mlp.net_with_heads
]
# flax.errors.ScopeParamNotFoundError: No parameter named "kernel" exists in "/layers_0/layers_0".


# another_mlp = networks.mlp(5, (8,))
# another_mlp.layers.append(nn.relu)
# another_mlp.layers.append(nn.Dense(features=2))
# another_mlp_params = another_mlp.init(next(rng), example_input)

# another_mlp.apply(another_mlp_params, test_input)


# shared_body = networks.Sequential([nn.Dense(features=6), nn.relu, nn.Dense(5), nn.relu])
# shared_params = shared_body.init(next(rng), example_input)

# head_input_shape = dense_output_shape(example_input.shape, 5)
# head_0 = nn.Dense(features=2)
# head_0_params = head_0.init(next(rng), jnp.zeros(head_input_shape))

# target = jrand.uniform(next(rng), example_input.shape)


# body_out = shared_body.apply(shared_params, test_input)

# loss, grads_head_0 = mse_grad_fn(head_0_params, body_out, target, head_0)


# from dopamine.discrete_domains import atari_lib

# dqn_input_shape = atari_lib.NATURE_DQN_OBSERVATION_SHAPE + (
#     atari_lib.NATURE_DQN_STACK_SIZE,
# )
# dqn = jax_networks.NatureDQNNetwork(num_actions=env.action_space.n)
# dqn_params = dqn.init(next(rng), jnp.zeros(dqn_input_shape))
