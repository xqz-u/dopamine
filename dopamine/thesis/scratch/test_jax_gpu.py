import time

import numpy as np
from jax import numpy as jnp
from jax import random as jrand


def timer(expr: callable):
    start = time.time()
    ret = expr()
    print(f"exec time: {time.time() - start}")
    return ret


key = jrand.PRNGKey(0)
x = jrand.normal(key, (10,))
print(x)

size = 3000
x = jrand.normal(key, (size, size), dtype=jnp.float32)
timer(lambda: jnp.dot(x, x.T).block_until_ready())


from jax import device_put

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
timer(lambda: jnp.dot(x, x.T).block_until_ready())
