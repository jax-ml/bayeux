# Copyright 2024 The bayeux Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imports from submodules."""
# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
import importlib.util

__all__ = []

if importlib.util.find_spec("jaxopt") is not None:
  from bayeux._src.optimize.jaxopt import BFGS
  from bayeux._src.optimize.jaxopt import GradientDescent
  from bayeux._src.optimize.jaxopt import LBFGS
  from bayeux._src.optimize.jaxopt import NonlinearCG
  __all__.extend(["BFGS", "GradientDescent", "LBFGS", "NonlinearCG"])

if importlib.util.find_spec("optax") is not None:
  from bayeux._src.optimize.optax import AdaBelief
  from bayeux._src.optimize.optax import Adafactor
  from bayeux._src.optimize.optax import Adagrad
  from bayeux._src.optimize.optax import Adam
  from bayeux._src.optimize.optax import Adamax
  from bayeux._src.optimize.optax import Adamw
  from bayeux._src.optimize.optax import Amsgrad
  # from bayeux._src.optimize.optax import Dpsgd  # pylint: disable=line-too-long
  from bayeux._src.optimize.optax import Fromage
  from bayeux._src.optimize.optax import Lamb
  # from bayeux._src.optimize.optax import Lars
  from bayeux._src.optimize.optax import Lion
  from bayeux._src.optimize.optax import NoisySgd
  from bayeux._src.optimize.optax import Novograd
  # from bayeux._src.optimize.optax import OptimisticGradientDescent  # pylint: disable=line-too-long
  from bayeux._src.optimize.optax import Radam
  from bayeux._src.optimize.optax import Rmsprop
  if importlib.util.find_spec("optax.contrib._schedule_free") is not None:
    from bayeux._src.optimize.optax import ScheduleFree
    __all__.append("ScheduleFree")
  from bayeux._src.optimize.optax import Sgd
  from bayeux._src.optimize.optax import Sm3
  from bayeux._src.optimize.optax import Yogi

  __all__.extend([
      "AdaBelief",
      "Adafactor",
      "Adagrad",
      "Adam",
      "Adamw",
      "Adamax",
      "Amsgrad",
      "Fromage",
      "Lamb",
      # "Lars",
      "Lion",
      "NoisySgd",
      "Novograd",
      # "OptimisticGradientDescent",
      # "Dpsgd",
      "Radam",
      "Rmsprop",
      "Sgd",
      "Sm3",
      "Yogi",
      ])

if importlib.util.find_spec("optimistix") is not None:
  from bayeux._src.optimize.optimistix import BFGS as optimistix_BFGS
  from bayeux._src.optimize.optimistix import NelderMead
  from bayeux._src.optimize.optimistix import NonlinearCG as optimistix_NonlinearCG

  __all__.extend([
      "optimistix_BFGS",
      "NelderMead",
      "optimistix_NonlinearCG"])
