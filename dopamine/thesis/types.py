from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import optax
from dopamine.discrete_domains import atari_lib, gym_lib
from flax import linen as nn
from flax.core import frozen_dict
from jax import numpy as jnp

from thesis import custom_pytrees

# a loss function, e.g. mse
LossMetric = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

# a nn.Module represented as a tree e.g.
# (networks.EnsembledNet,
#  {"model": (networks.MLP,
#             {"features": 2, "hiddens": (4,)}),
#   "n_heads": 4})
ModelDef = Tuple[Callable[..., nn.Module], Dict[str, Union[Any, "ModelDef"]]]

# the complete ingredients to define a TrainState: the function
# approximator spec, an optimizer and a loss function, e.g.
# ((networks.EnsembledNet,
#   {"model": (networks.MLP,
#              {"features": 2, "hiddens": (4,)}),
#    "n_heads": 4}),
#  optax.sgd(0.1),
#  losses.mse_loss)
# NOTE changed; call agent.utils.build_models(ModelTSDef[0]) first to
# have the below typesig object
# TODO create a namedtuple for this!
ModelTSDef = Tuple[nn.Module, optax.GradientTransformation, LossMetric]

# shorter type alias for a nn.Module forward pass call
ModuleCall = Callable[[frozen_dict.FrozenDict, jnp.ndarray], jnp.ndarray]

# the type of information returned by a training/evaluation routine
MetricValue = Union[jnp.ndarray, np.ndarray, float]
MetricsDict = Dict[str, Union[MetricValue, "MetricsDict"]]

# the __call__ method of a thesis.exploration.base.PolicyEvaluation
# object returns the RNG updated after its usage, the chosen action, and
# (possibly empty) dictionary of additional info defined in the object
# itself
PolicyEvalInfo = Tuple[custom_pytrees.PRNGKeyWrap, jnp.ndarray, MetricsDict]

# tyes of discrete environments supported
DiscreteEnv = Union[gym_lib.GymPreprocessing, atari_lib.AtariPreprocessing]
