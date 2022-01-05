from typing import Optional

import gin
import optax


@gin.configurable
def adam_optimizer(
    learning_rate: float = 6.25e-5,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-08,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)


@gin.configurable
def rmsprop_optimizer(
    learning_rate: float = 6.25e-5,
    decay: float = 0.999,
    eps: float = 1.5e-4,
    centered: bool = False,
) -> optax.GradientTransformation:
    return optax.rmsprop(learning_rate, decay=decay, eps=eps, centered=centered)


@gin.configurable
def sgd_optimizer(
    learning_rate: float = 6.25e-5,
    momentum: Optional[float] = None,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    return optax.sgd(learning_rate, momentum, nesterov)
