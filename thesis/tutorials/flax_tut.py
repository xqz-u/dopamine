# https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html

from typing import Callable, Sequence

import jax
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax import lax
from jax import numpy as jnp
from jax import random as jrand

# ------------------------- Flax Linear Regression -------------------------


# We create one dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)

# Parameters are not stored with the models themselves. You need to initialize
# parameters by calling the init function, using a PRNGKey and a dummy input
# parameter.
key1, key2 = jrand.split(jrand.PRNGKey(0))
x = jrand.normal(key1, (10,))  # Dummy input
# NOTE model definition requires output space definition, input space size is
# inferred in the follwing way
params = model.init(key2, x)  # Initialization call
# model: R^10 -> R^5
jax.tree_map(lambda x: x.shape, params)  # Checking output shapes

# NOTE params are immutable!
try:
    params["new_key"] = jnp.ones((2, 2))
except ValueError as e:
    print("Error: ", e)


# To evaluate the model with a given set of parameters (never stored with the
# model), we just use the apply method by providing it the parameters to use as
# well as the input:
model.apply(params, x)


# GRADIENT DESCENT
# Set problem dimensions
nsamples = 20
xdim = 10
ydim = 5

# Generate random ground truth W and b
key = jrand.PRNGKey(0)
k1, k2 = jrand.split(key)
W = jrand.normal(k1, (xdim, ydim))
b = jrand.normal(k2, (ydim,))
true_params = freeze({"params": {"bias": b, "kernel": W}})

# Generate samples with additional noise
ksample, knoise = jrand.split(k1)
x_samples = jrand.normal(ksample, (nsamples, xdim))
y_samples = jnp.dot(x_samples, W) + b
y_samples += 0.1 * jrand.normal(knoise, (nsamples, ydim))  # Adding noise
print("x shape:", x_samples.shape, "; y shape:", y_samples.shape)


# loss function
def make_mse_func(x_batched, y_batched):
    def mse(params):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y - pred, y - pred) / 2.0

        # We vectorize the previous to compute the average of the loss on all
        # samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

    return jax.jit(mse)  # And finally we jit the result.


# Get the sampled loss
loss = make_mse_func(x_samples, y_samples)

# perform gradient descent optimization
alpha = 0.3  # Gradient step size
print('Loss for "true" W,b: ', loss(true_params))
grad_fn = jax.value_and_grad(loss)

for i in range(101):
    # We perform one gradient update
    loss_val, grads = grad_fn(params)
    params = jax.tree_multimap(lambda p, g: p - alpha * g, params, grads)
    if i % 10 == 0:
        print("Loss step {}: ".format(i), loss_val)


# ------------------------- Optimization -------------------------
# above sgd using library functions with Optax (from DeepMind)

# optimization process:
# - Choose an optimization method (e.g. optax.sgd).
# - Create optimizer state from parameters.
# - Compute the gradients of your loss with jax.value_and_grad().
# - At every iteration, call the Optax update function to update the internal
#   optimizer state and create an update to the parameters. Then add the update
#   to the parameters with Optax’s apply_updates method.

import optax

tx = optax.sgd(learning_rate=alpha)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss)


for i in range(101):
    loss_val, grads = loss_grad_fn(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print("Loss step {}: ".format(i), loss_val)


# ------------------------- Serialization -------------------------
from flax import serialization

bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print("Dict output")
print(dict_output)
print("Bytes output")
print(bytes_output)


# To load the model back, you’ll need to use as a template the model parameter
# structure, like the one you would get from the model initialization. Here, we
# use the previously generated params as a template. Note that this will
# produce a new variable structure, and not mutate in-place.

# NOTE then how would I go from serializing the model to reading it back? One
# always needs to have the exact model definition (params), then the optimized
# params can be read back
serialization.from_bytes(params, bytes_output)


# and now the important part...
# ------------------------- Custom model definition -------------------------

# The base abstraction for models is the nn.Module class. Define a simple MLP
# NOTE nn.Module is a dataclass
class ExplicitMLP(nn.Module):
    features: Sequence[int]

    # setup here needed stuff
    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    # then a ExplicitMLP instance will call .apply, which wraps a call to this
    def __call__(self, inputs):
        # NOTE acts as ft.reduce, maybe it is better not to use that api here
        # though...
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


key1, key2 = jrand.split(jrand.PRNGKey(0))
x = jrand.uniform(key1, (4, 4))

# NOTE features always define the output size of a layer and its bias
# dimensionality, and the length of the features is the number of layers in the
# NN. Initializing the model will finalize a layer's input dimensionality.
# *QUESTION* how is weights initiazation controlled? eg should I use dirichlet
# distribution or other functions on the dummy input cause this is the place
# for weight initialization, or is it done in some other way?
model = ExplicitMLP(features=[3, 4, 5])
params = model.init(key2, x)
jax.tree_map(lambda x: x.shape, params)
y = model.apply(params, x)

print("initialized parameter shapes:\n", jax.tree_map(jnp.shape, unfreeze(params)))
print("output:\n", y)


# Alternative module declaration
# NOTE this is not explained as thouroughly in the docs, and it looks a bit
# experimental plus it has limitations (iiuc, new submodules like the Dense
# ones can only be declared in the nn.compact annotated function, and you can
# only annotate one Module method). To be on the safe side, just use the .setup
# method, and remember to assign to self whatever you will need later (eg don't
# do self.layers.append but self.layers = [...])
# NOTE also that it is not crucial to avoid layers recreation in a nn.Module, since
# the parameters of the network (weights and bias) are separated from its
# architecture! if layer creation is cheap, then it is not a problem to keep
# creating them in a nn.compact annotated __call__


class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
            # providing a name is optional though!
            # the default autonames would be "Dense_0", "Dense_1", ...
        return x


key1, key2 = jrand.split(jrand.PRNGKey(0), 2)
x = jrand.uniform(key1, (4, 4))

model = SimpleMLP(features=[3, 4, 5])
params = model.init(key2, x)
y = model.apply(params, x)

print("initialized parameter shapes:\n", jax.tree_map(jnp.shape, unfreeze(params)))
print("output:\n", y)


# ------------------------- Module parameters -------------------------

# Create a Dense layer if it was not already provided
# NOTE this example provides an answer to the question of weight and bias
# initialization above for layers, which btw have proper keyword args to change
# default initializers
class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    # NOTE use nn.Module.param function to assign params to a model, can also
    # be used in .setup
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel",
            self.kernel_init,  # Initialization function
            (inputs.shape[-1], self.features),
        )  # shape info.
        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
        )  # TODO Why not jnp.dot? NOTE indeed
        bias = self.param("bias", self.bias_init, (self.features,))
        y = y + bias
        return y


key1, key2 = jrand.split(jrand.PRNGKey(0))
x = jrand.uniform(key1, (4, 4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print("initialized parameters:\n", params)
print("output:\n", y)


# ------------------------- Variables and collections of variables -------------------------

# Working with models means working with:
# - A subclass of nn.Module (architecture)
# - A pytree of parameters for the model (typically from model.init()) (NN
#   state)
# This is not enough for NN though, it does not include keeping a state for eg
# running averages or such required by a batch normalization (NOTE they could
# probably be kept outside the network and updated with the external params? Or
# are there more restrictions, besides this vanilla solution being messy?)


# implement a simplified but similar mechanism to batch normalization
class BiasAdderWithRunningMean(nn.Module):
    decay: float = 0.99

    @nn.compact
    def __call__(self, x):
        # easy pattern to detect if we're initializing via empty variable tree
        is_initialized = self.has_variable("batch_stats", "mean")
        ra_mean = self.variable(
            "batch_stats", "mean", lambda s: jnp.zeros(s), x.shape[1:]
        )
        mean = ra_mean.value  # This will either get the value or trigger init
        bias = self.param("bias", lambda rng, shape: jnp.zeros(shape), x.shape[1:])
        if is_initialized:
            ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(
                x, axis=0, keepdims=True
            )

        return x - ra_mean.value + bias


key1, key2 = jrand.split(jrand.PRNGKey(0))
x = jnp.ones((10, 5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print("initialized variables:\n", variables)
y, updated_state = model.apply(variables, x, mutable=["batch_stats"])
print("updated state:\n", updated_state)


# To update the variables and get the new parameters of the model, we can use
# the following pattern:
for val in [1.0, 2.0, 3.0]:
    x = val * jnp.ones((10, 5))
    y, updated_state = model.apply(variables, x, mutable=["batch_stats"])
    old_state, params = variables.pop("params")
    variables = freeze({"params": params, **updated_state})
    print("updated state:\n", updated_state)  # Shows only the mutable part

# NOTE what I do not get is why do we need nn.Module.param and
# nn.Module.variable, and I think it is because in this way they can become
# part of the architecture definition. Moroever this holds for .variable, which
# acts as mutable when it actually is not
# NOTE from the docs: From this simplified example, you should be able to
# derive a full BatchNorm implementation, or any layer involving a state.


# final example complete training routine for BiasAdderWithRunningMean NOTE the
# docs say this would not be jittable
def update_step(tx, apply_fn, x, opt_state, params, state):
    def loss(params):
        y, updated_state = apply_fn(
            {"params": params, **state}, x, mutable=list(state.keys())
        )
        l = ((x - y) ** 2).sum()
        return l, updated_state

    (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, state


x = jnp.ones((10, 5))
variables = model.init(jrand.PRNGKey(0), x)
state, params = variables.pop("params")
del variables
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for _ in range(3):
    opt_state, params, state = update_step(tx, model.apply, x, opt_state, params, state)
    print("Updated state: ", state)
