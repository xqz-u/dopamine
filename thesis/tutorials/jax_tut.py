# https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html
import jax
import numpy as np  # We import the standard NumPy library
from jax import numpy as jnp
from jax import random

from thesis import utils as u

# ------------------------- DeviceArray -------------------------

# We're generating one 4 by 4 matrix filled with ones.
m = jnp.ones((3, 2))
# An explicit 2 by 4 array
n = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
m
n

# vector multiplication NOTE remember:
# i x j
# j x k
# -> i x k

# same result as np.dot(m)
jnp.dot(m, n)
# another syntax from numpy:
m @ n
# NOTE DeviceArray instances behave as _futures_ !!!
jnp.dot(m, n).block_until_ready()


a = jnp.array([[1, 2], [3, 4]])
b = jnp.array([[11, 12], [13, 14]])
print(a)
print(b)
print(jnp.dot(a, b))
print(jnp.inner(a, b))


# JAX DeviceArray are fully compatible with NumPy's ones
x = np.random.normal(size=(1, 3))
x
x @ m
jnp.dot(x, m)


# computation between jnp arrays lives on the selected device (gpu, tpu, cpu) and
# there is no intermittent transfer to memory (at least I think this is what this
# means). alternatively, it might mean that in a distributed computining
# environment, the actual data leave the host of computation to reach another
# node only when requested?
# explicitly call this to move e.g. a np array to the selected device, or create
# a DeviceArray directly
x = np.array(x)
jax.device_put(x)


# ------------------------- Immutability -------------------------

x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
try:
    x[0][0] = 2
except TypeError:
    print("jnp arrays are immutable, no in-place modification")
updated = jax.ops.index_update(x, (0, 0), 3.0)  # whereas x[0,0] = 3.0 would fail
updated
id(updated) is not id(x)
updated = updated.at[0, 1].set(6)
updated

# index operations are found in jax.ops.index*
x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
x
p = jax.ops.index_update(x, jax.ops.index[0, :], 3.0)  # Same as x[O,:] = 3.0 in NumPy.
jax.ops.index_update(x, jax.ops.index[0, :], 3.0)  # Same as x[O,:] = 3.0 in NumPy.
# or use more concise wrapper .at DeviceArray method
# bottom line is, modified array is new arry, no in-place modification
x.at[0, :].set(3)
index = jax.ops.index[0, :]
x.at[index]
x.at[index].set(3)
# other operations following this syntax
x.at[0, :].mul(2)
x.at[0, 0].max(10)
x.at[0, :].min(jnp.zeros(x.shape[1]))
# operations are ofc vectorized...
x.at[0, :].min(x[0, :] - 1)


# ------------------------- Randomness -------------------------
# from the docs:
# JAX implements an explicit PRNG where entropy production and consumption are
# handled by explicitly passing and iterating a PRNG state. JAX uses a modern
# Threefry counter-based PRNG that’s splittable. That is, its design allows us
# to fork the PRNG state into new PRNGs for use with parallel stochastic
# generation.

# the state of the PRNG is represented by two int32, which together stand for a
# uint64; it is called a key
key = random.PRNGKey(0)
key

# NOTE the same PRNG always produces the same output!
for i in range(3):
    print(
        "Printing the random number using key: ",
        key,
        " gives: ",
        random.normal(key, shape=(1,)),
    )

# NOTE spltting a key changes the previous PRNG state?!
print("old key", key, "--> normal", random.normal(key, shape=(1,)))
key, subkey = random.split(key)
key
subkey
print("    \---SPLIT --> new key   ", key, "--> normal", random.normal(key, shape=(1,)))
print(
    "             \--> new subkey",
    subkey,
    "--> normal",
    random.normal(subkey, shape=(1,)),
)

# can also specify number of subkeys
# TODO Question : why is there no overlapping result with random.split(key) (so only one number generated with the same key)
random.split(key, 4)
# note the interesting destructuring in car cdr on the lhs, did not know this!
key, *subkeys = random.split(key, 4)
key
subkeys

# NOTE bottom line is, to make n calls to produce random values, n keys are needed!
# JAX uses this tree structures for PRNG states to make experiments easily
# reproducible: the new PRNG states resulting from a split are deterministic!
for _ in range(3):
    key = random.PRNGKey(0)
    print(key)
    key, subkey = random.split(key)
    print(key)
    print(subkey)

# ------------------------- Gradients -------------------------
# first-class support for gradients and automatic differentiation in functions
# TODO see more at https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

key = random.PRNGKey(0)


def f(x):
    return jnp.dot(x.T, x) / 2.0


v = jnp.ones((4,))
v
f(v)

v = random.normal(key, (4,))
print(v)
print(f(v))
grad_f = jax.grad(f)
# NOTE this returns the input because of the specific definition of f...
print(grad_f(v))

# other points deal with Jacobian vector product (jax.jvp) and vector Jacobian
# product (jax.vjp), where the latter is used to compute gradients in backprop I
# think. skip exploring these for the moment, look into them more in the advanced
# guide linked above

# ------------------------- jit & ops vectorizatio -------------------------
# optimization time


def selu(x, alpha=1.67, _lambda=1.05):
    return _lambda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


v = random.normal(key, (1000000,))
v
v.shape


@u.timer
def t():
    selu(v).block_until_ready()


t()

selu_jit = jax.jit(selu)


@u.timer
def tjit():
    selu_jit(v).block_until_ready()


# 10x speedup
tjit()

# vectorization


batched_x = random.normal(key, (5, 10))  # Batching on first dimension
single = random.normal(key, (10,))
mat = random.normal(key, (15, 10))


def apply_matrix(v):
    return jnp.dot(mat, v)


print("Single apply shape: ", apply_matrix(single).shape)
print("Batched example shape: ", jax.vmap(apply_matrix)(batched_x).shape)


# ------------------------- Full example: Linear Regression -------------------------
key = random.PRNGKey(0)

# Create the predict function from a set of parameters
def make_predict(W, b):
    def predict(x):
        return jnp.dot(W, x) + b

    return predict


# Create the loss from the data points set
def make_mse(x_batched, y_batched):
    def mse(W, b):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            y_pred = make_predict(W, b)(x)
            # NOTE should be able to substitute with jnp.dot in the 1D case like
            # this one
            return jnp.inner(y - y_pred, y - y_pred) / 2.0

        # We vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

    return jax.jit(mse)  # And finally we jit the result.


# Set problem dimensions
nsamples = 20
xdim = 10
ydim = 5

# Generate random ground truth W and b
k1, k2 = random.split(key)
W = random.normal(k1, (ydim, xdim))
b = random.normal(k2, (ydim,))
true_predict = make_predict(W, b)

# Generate samples with additional noise
ksample, knoise = random.split(k1)
# input values used to compute y = f(x), where f is the ground truth model
x_samples = random.normal(ksample, (nsamples, xdim))
# truth values we use for modelling, created by adding noise to the true
# underlying model's prediction
y_samples = jax.vmap(true_predict)(x_samples) + 0.1 * random.normal(
    knoise, (nsamples, ydim)
)

# Generate MSE for our samples
mse = make_mse(x_samples, y_samples)


# Initialize estimated W and b with zeros.
What = jnp.zeros_like(W)
bhat = jnp.zeros_like(b)

alpha = 0.3  # Gradient step size NOTE learning rate?
# should be 0
print('Loss for "true" W,b: ', mse(W, b))
# NOTE is this SGD?
for i in range(101):
    # We perform one gradient update
    # NOTE that the gradient is taken wrt positionanl argument of `mse` to
    # actually compute partial derivatives; the parameters update is fully
    # explicit because this model only has 2 parameters, should be possible to
    # replace it with a loop wich takes the partial derivatives wrt to the
    # differentiated function's number of arguments
    What -= alpha * jax.grad(mse, 0)(What, bhat)
    bhat -= alpha * jax.grad(mse, 1)(What, bhat)
    if i % 5 == 0:
        print("Loss step {}: ".format(i), mse(What, bhat))


# ------------------------- pytrees (JAX internal data structures) -------------------------

# In JAX, a pytree is a container of leaf elements and/or more pytrees.
# Containers include lists, tuples, and dicts (JAX can be extended to consider
# other container types as pytrees, see Extending pytrees below). A leaf
# element is anything that’s not a pytree, e.g. an array. In other words, a
# pytree is just a possibly-nested standard or user-registered Python
# container. If nested, note that the container types do not need to match. A
# single “leaf”, i.e. a non-container object, is also considered a pytree.

[1, "a", object()]  # 3 leaves: 1, "a" and object()

(1, (2, 3), ())  # 3 leaves: 1, 2 and 3

[1, {"k1": 2, "k2": (3, 4)}, 5]  # 5 leaves: 1, 2, 3, 4, 5

from jax import tree_util

t = [1, {"k1": 2, "k2": (3, 4)}, 5]
t

tree_util.tree_map(lambda x: x * x, t)
# a bit like mapping over multiple sequences in lisp, such that args are
# extracted in zip order
tree_util.tree_multimap(lambda x, y: x + y, t, tree_util.tree_map(lambda x: x * x, t))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LINEAR REGRESSION WITH PYTREES

key = random.PRNGKey(0)

# Create the predict function from a set of parameters
def make_predict_pytree(params):
    def predict(x):
        return jnp.dot(params["W"], x) + params["b"]

    return predict


# Create the loss from the data points set
def make_mse_pytree(x_batched, y_batched):
    def mse(params):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            y_pred = make_predict_pytree(params)(x)
            return jnp.inner(y - y_pred, y - y_pred) / 2.0

        # We vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

    return jax.jit(mse)  # And finally we jit the result.


# Generate MSE for our samples
mse_pytree = make_mse_pytree(x_samples, y_samples)
# Initialize estimated W and b with zeros.
params = {"W": jnp.zeros_like(W), "b": jnp.zeros_like(b)}

# NOTE important difference: jax.grad can take gradients with respect to pytrees,
# so that taking the gradient wrt to a pytree (note all the partial derivatives)
# behaves like a first-order citizen! useful for functions with many
# parameters, eg NNs
jax.grad(mse_pytree)(params)


print('Loss for "true" W,b: ', mse_pytree({"W": W, "b": b}))
for i in range(101):
    # We perform one gradient update
    # NOTE the beauty and conciseness once this concept of pytrees is well
    # defined, enabling to use `tree_multimap`
    params = jax.tree_multimap(
        lambda old, grad: old - alpha * grad, params, jax.grad(mse_pytree)(params)
    )
    if i % 5 == 0:
        print("Loss step {}: ".format(i), mse_pytree(params))
