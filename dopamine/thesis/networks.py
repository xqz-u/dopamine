from typing import Sequence, Tuple, Union

from flax import linen as nn
from jax import numpy as jnp


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x) -> jnp.DeviceArray:
        for layer in self.layers:
            x = layer(x)
        return x

    # TODO use some library function or such!
    def __hash__(self) -> int:
        return hash(".".join(str(hash(l)) for l in self.layers))


# NOTE setting _min_vals and _max_vals because of
# https://flax.readthedocs.io/en/latest/flax.errors.html#flax.errors.SetAttributeInModuleSetupError
class DensePreproc(nn.Dense):
    min_vals: Union[None, Tuple[float, ...]] = None
    max_vals: Union[None, Tuple[float, ...]] = None
    inputs_preprocessed: bool = False

    def setup(self):
        if self.min_vals is not None:
            assert self.max_vals is not None
            self._min_vals = jnp.array(self.min_vals)
            self._max_vals = jnp.array(self.max_vals)

    def __call__(self, x) -> jnp.DeviceArray:
        if not self.inputs_preprocessed:
            x = x.reshape((-1))  # flatten
            if self.min_vals is not None:
                x -= self._min_vals
                x /= self._max_vals - self._min_vals
                x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        return super().__call__(x)


def mlp(
    output_dim: int,
    hiddens: Sequence[int] = (64, 64),
    activation_fn: callable = nn.relu,
    **dense_kwargs
) -> Sequential:
    return Sequential(
        [DensePreproc(features=hiddens[0], **dense_kwargs)]
        + [
            v
            for tup in zip(
                [activation_fn] * len(hiddens),
                [
                    DensePreproc(features=out_feat)
                    for out_feat in hiddens[1:] + (output_dim,)
                ],
            )
            for v in tup
        ],
    )
