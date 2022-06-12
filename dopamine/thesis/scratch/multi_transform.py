# https://github.com/google/flax/discussions/1453#discussioncomment-2592634
import functools as ft

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import frozen_dict


class Model(nn.Module):
    num_layers: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = nn.relu(nn.Dense(self.depth)(x))
        return nn.Dense(self.depth, name="head")(x)


model = Model(num_layers=3, depth=10)
x = jnp.zeros([1, 7])
params = model.init(jax.random.PRNGKey(0), x)["params"]

jax.tree_map(jnp.shape, params)

fake_grads = jax.tree_map(jnp.ones_like, params.unfreeze())


def flattened_traversal(fn):
    """
    Returns function that is called with `(path, param)` instead of
    pytree.
    """

    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    # recursive version; it won't work for this example operation
    # though. should either change the traversal not to go all the the
    # way down to the leaves, or the mappend fn

    # def mask(tree):
    #     print(tree.keys())
    #     return {
    #         k: (mask(v) if isinstance(v, dict) else fn(k, v)) for k, v in tree.items()
    #     }

    return mask


def fn(path, what):
    print(path)
    print(what)
    print("_________")
    return "sgd" if path[0] == "head" else "none"


# Freezes all but the last layer.
label_fn = flattened_traversal(fn)
tx = optax.multi_transform(
    {"sgd": optax.sgd(0.1), "none": optax.set_to_zero()}, label_fn
)

opt_state = tx.init(params.unfreeze())
updates, opt_state = tx.update(fake_grads, opt_state)

jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), updates)


# ---------------------------------


# 2 ways of partitioning the trainable and non trainable parameters:


# A
# with a function that, walking the parameters tree, returns a bool
# indicating whether the corresponding transform should be applied
def labeller(head_suffix, keys, _):
    return "sgd" if keys[0] == f"Dense_{head_suffix}" else "none"


# the function always takes (collection_keys, values)
labeller_dense_0 = ft.partial(labeller, 0)

label_fn = flattened_traversal(labeller_dense_0)
tx = optax.multi_transform(
    {"sgd": optax.sgd(0.1), "none": optax.set_to_zero()}, label_fn
)
opt_state = tx.init(params.unfreeze())
updates, opt_state = tx.update(fake_grads, opt_state)

# B
# or with a PyTree with the same structure of parameters - or just a
# prefix of them
tx = optax.multi_transform(
    {"sgd": optax.sgd(0.1), "none": optax.set_to_zero()},
    frozen_dict.freeze({"Dense_0": "sgd", "Dense_1": "none", "head": "none"}),
)
opt_state = tx.init(params)
# NOTE when using flax.training.train_state.TrainState, type matching is
# already implemented - frozen mask with frozen grads, same for
# unfrozen, but both have to be the same; see
# https://github.com/google/flax/discussions/1706#discussioncomment-1784119
updates, opt_state = tx.update(frozen_dict.freeze(fake_grads), opt_state)
