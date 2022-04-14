import timeit

code = """
import numpy as np
from jax import device_put
from jax import numpy as jnp
from jax import random as jrand


key = jrand.PRNGKey(0)
size = 3000
x = jrand.normal(key, (size, size), dtype=jnp.float32)
y = np.random.normal(size=(size, size)).astype(np.float32)
y = device_put(y)
"""
# stern: 32.80970892100595
# peregrine: 8.99069713299832
# peregrine gpu: 1.5610818099230528
exec_time = timeit.timeit("jnp.dot(x, x.T).block_until_ready()", setup=code, number=100)
print(f"exec time: {exec_time}")

# stern: 32.76028992999636
# peregrine: 9.075912225998763
# peregrine gpu: 0.6199215138331056
exec_time = timeit.timeit("jnp.dot(y, y.T).block_until_ready()", setup=code, number=100)
print(f"exec time: {exec_time}")
