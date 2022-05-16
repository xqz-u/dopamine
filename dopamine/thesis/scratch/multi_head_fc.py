from typing import List, Tuple, Union

import gym
import jax
import numpy as np
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
params = mlp.init(next(rng), example_input)
utils.jax_container_shapes(params)


from dataclasses import field


# wrong: parameters of each dense head are the same! initilization of
# each head should be controlled by me!
class MultiHead(nn.Module):
    features: int
    n_heads: int
    dense_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x):
        return [
            networks.DensePreproc(features=self.features, **self.dense_kwargs)(x)
        ] * self.n_heads


net = MultiHead(features=2, n_heads=2)
params = net.init(next(rng), example_input)
utils.jax_container_shapes(params)
net.apply(params, test_input)


def pippo(rng, model, input):
    return model.init(rng, input)


# NOTE this is correct, but you can vmap only on np/jnp arrays, so with
# numeric types (I believe)
def initter(rng, input):
    mods = [networks.DensePreproc(features=2)] * 2
    return jax.vmap(pippo, in_axes=(0, 0, None))(jrand.split(rng), mods, input)


# el = initter(next(rng), example_input)


class MHD(nn.Module):
    @nn.compact
    def __call__(self, x):
        heads = [networks.DensePreproc(features=2)] * 2
        init_fn = lambda rng, xx: jax.vmap(lambda h: h.init, in_axes=(0, 0))(
            heads, jrand.split(rng, 2), x
        )
        # ["params"]
        apply_fn = jax.vmap(lambda h: h.apply, in_axes=(0, 0))
        mhd_params = self.param("mhd", init_fn, x)
        return apply_fn({"params": mhd_params}, x)


net = MHD()
# params = net.init(next(rng), example_input)
# utils.jax_container_shapes(params)
# net.apply(params, test_input)


class MultiHeadDense(nn.Module):
    features: int
    n_heads: int
    rng: custom_pytrees.PRNGKeyWrap
    dense_kwargs: dict = {}

    def setup(self):
        self.heads: List[Tuple[networks.DensePreproc, FrozenDict]] = []
        for _ in self.n_heads:
            head = networks.DensePreproc(features=self.features, **self.dense_kwargs)
            head_params = head.init()

        # self.heads: List[networks.DensePreproc] = [
        #     networks.DensePreproc(**self.heads_kwargs)
        # ] * self.n_heads
        # for

        heads_params: List[FrozenDict]

    def __call__(self, x: Union[jnp.DeviceArray, np.ndarray]) -> jnp.DeviceArray:
        # res = []
        # for head in self.heads:
        #     ret = head(x)
        #     print(ret)
        #     res.append(ret)
        # return res
        return jax.vmap(lambda head: head(x))(self.heads)


# mh_net = MultiHeadDense(features = 2, n_heads = 10, rng = rng, dense_kwargs={})

# mhfc = MultiHeadDense(n_heads=2, heads_kwargs={"features": env.action_space.n})
# mhfc_params = mhfc.init(next(rng), example_input)

# preds = mhfc.apply(mhfc_params, test_input)


class Pippo(nn.Module):
    def setup(self):
        self.heads = [nn.Dense(features=2, name=f"dense_{i}") for i in range(2)]

    def initter(self, rng, x):
        rngs = jrand.split(rng, 2)
        return [h.init(r, x) for r, h in zip(rngs, self.heads)]

    def __call__(self, x, rng=None):
        ...


el = Pippo()
p = el.init(next(rng), example_input, method=Pippo.initter)


# class MultiHeadDense:
#     features: int
#     n_heads: int

#     def __init__(self):
#         ...


class ManualDense:
    rng: custom_pytrees.PRNGKeyWrap
    fc: nn.Dense
    params: FrozenDict
    act: callable

    def __init__(self, features, input_shape):
        self.rng = custom_pytrees.PRNGKeyWrap(0)
        self.fc = nn.Dense(features=features)
        self.params = self.fc.init(next(rng), jnp.ones(input_shape))

    def __call__(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        return self.fc.apply(self.params, x)


def mse(params, x_batched, y_batched, model):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


mse_grad_fn = jax.value_and_grad(mse)

net = ManualDense(2, env_spec["observation_shape"])
target = rand_input(shape=(4, 2))
pred = net(example_input)

loss, grads = mse_grad_fn(net.params, example_input, target, net.fc)


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


# from thesis.agents import agent_utils
import optax


# TODO find interface thas can be written agnostically, passing all the
# requested parameters - e.g. loss_metric is missing rn, activation_fn
# NOTE would be cool to be able to pass this class to jitted functions!
# for now just use a for loop, since you can still pass
# NetworkOptimWraps...
class EnsembledNet:
    net_with_heads: List[custom_pytrees.NetworkOptimWrap] = []

    def __init__(
        self,
        n_heads: int,
        shared_body: nn.Module,
        features: int,
        ex_input: jnp.DeviceArray,
        rng: custom_pytrees.PRNGKeyWrap,
    ):
        shared_body = networks.Sequential([shared_body, nn.relu])
        # shared_params = shared_body.init(next(rng), ex_input)
        for i in range(n_heads):
            shared_body.layers.append(nn.Dense(features=features))
            # head =
            mutable_params = shared_params.unfreeze()
            mutable_params["params"][f"head_{i}"] = head.init(next(rng), ex_input)
            shared_params = FrozenDict(mutable_params)
            opt = optax.adam(learning_rate=0.001, eps=3.125e-4)
            opt_state = opt.init(shared_params)
            self.net_with_heads.append(
                custom_pytrees.NetworkOptimWrap(
                    shared_params,
                    opt_state,
                    networks.Sequential([shared_body.layers, head]),
                    opt,
                )
            )


ensemble_mlp = EnsembledNet(3, mlp, env.action_space.n, example_input, rng)


def dense_output_shape(in_shape: tuple, out_features: int) -> tuple:
    return (in_shape[0], out_features)


shared_body = networks.Sequential([nn.Dense(features=6), nn.relu, nn.Dense(5), nn.relu])
shared_params = shared_body.init(next(rng), example_input)

head_input_shape = dense_output_shape(example_input.shape, 5)
head_0 = nn.Dense(features=2)
head_0_params = head_0.init(next(rng), jnp.zeros(head_input_shape))

target = jrand.uniform(next(rng), example_input.shape)


body_out = shared_body.apply(shared_params, test_input)

loss, grads_head_0 = mse_grad_fn(head_0_params, body_out, target, head_0)
