from flax import linen as nn
from jax import random as jrand

shape_1 = (4, 1, 1)
shape_2 = (4, 1)
out_shape = 2

key = jrand.PRNGKey(seed=47)
key, key1 = jrand.split(key)

dense = nn.Dense(features=out_shape)

state1 = jrand.uniform(key1, shape_1)
state2 = state1.reshape(shape_2)

params = dense.init(key, state1)

dense.apply(params, state1)
dense.apply(params, state2)

# the network does not complain about the different shapes; this
# should be because both last dimensions have same shape, and from
# Dense's doc: Applies a linear transformation to the inputs along the
# last dimension. Therefore, also the output shapes match the input ones
