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

"""Shared functions for optimizers."""
import collections

from bayeux._src import debug
from bayeux._src import shared


OptimizerResults = collections.namedtuple("OptimizerResults",
                                          ["params", "state", "loss"])


def get_extra_kwargs(kwargs):
  return {
      "chain_method": kwargs.get("chain_method", "vectorized"),
      "num_particles": kwargs.get("num_particles", 8),
      "num_iters": kwargs.get("num_iters", 1000),
      "apply_transform": kwargs.get("apply_transform", True),
  }


def get_optimizer_kwargs(optimizer, kwargs, ignore_required=None):
  """Sets defaults and merges user-provided adaptation keywords."""
  if ignore_required is None:
    ignore_required = set()
  optimizer_kwargs, optimizer_required = shared.get_default_signature(
      optimizer)
  optimizer_kwargs.update(
      {k: kwargs[k] for k in optimizer_required if k in kwargs})

  optimizer_required = (optimizer_required -
                        ignore_required -
                        set(optimizer_kwargs.keys()))

  if optimizer_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(optimizer_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  optimizer_kwargs.update(
      {k: kwargs[k] for k in optimizer_kwargs if k in kwargs})
  return optimizer_kwargs


class Optimizer(shared.Base):
  """Base class for optimizers."""
  name: str = ""
  optimizer: str = ""

  def debug(
      self,
      seed,
      verbosity=2,
      catch_exceptions: bool = True,
      printer=print,
      kwargs=None,
  ):
    return debug.debug_no_ildj(
        self,
        seed,
        verbosity=verbosity,
        catch_exceptions=catch_exceptions,
        printer=printer,
        kwargs=kwargs,
    )

  def default_kwargs(self):
    return {"learning_rate": 0.1}

  def negative_log_prob(self):
    return lambda x: -self.log_density(x)

  def transformed_negative_log_prob(self):
    return lambda x: -self.log_density(self.transform_fn(x))

  def _map_optimizer(self, chain_method, fit):
    return shared.map_fn(chain_method, fit)

  def _prep_args(self, seed, kwargs):
    num_particles = kwargs["extra_parameters"]["num_particles"]
    initial_state = self.get_initial_state(seed, num_chains=num_particles)
    apply_transform = kwargs["extra_parameters"]["apply_transform"]
    if apply_transform:
      fun = self.transformed_negative_log_prob()
      initial_state = self.inverse_transform_fn(initial_state)
    else:
      fun = self.negative_log_prob()
    return fun, initial_state, apply_transform
