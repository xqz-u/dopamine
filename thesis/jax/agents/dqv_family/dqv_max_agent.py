#!/usr/bin/env python3

import attr
import gin
from flax.core.frozen_dict import FrozenDict


# TODO base class for DQV family
@gin.configurable
@attr.s(auto_attribs=True)
class JaxDQVMaxAgent:
    Q_online: FrozenDict = FrozenDict()
    Q_target: FrozenDict = FrozenDict()
    V_online: FrozenDict = FrozenDict()
