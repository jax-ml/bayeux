# Copyright 2023 The bayeux Authors.
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

"""API entry point."""

import dataclasses

from bayeux import mcmc
from bayeux import optimize
from bayeux import vi
from bayeux._src import shared
import oryx

_MODULES = (mcmc, optimize, vi)
_REQUIRED_KWARGS = (
    "log_density",
    "test_point",
    "transform_fn",
    "inverse_transform_fn",
    "inverse_log_det_jacobian",
    "initial_state",)


class _Namespace:

  def __init__(self):
    self._fns = []

  def __repr__(self):
    return "\n".join(self._fns)

  def __setclass__(self, clas, parent):
    kwargs = {k: getattr(parent, k) for k in _REQUIRED_KWARGS}
    setattr(self, clas.name, clas(**kwargs))
    self._fns.append(clas.name)


def is_tfp_bijector(bij):
  return hasattr(bij, "__class__") and bij.__class__.__module__.startswith(
      "tensorflow_probability")


@dataclasses.dataclass
class Model(shared.Base):
  """Base model class.

  Models consist of a possibly unnormalized log density function, a point that
  the function can act on, in that `log_density(test_point)` returns a scalar,
  and an optional `transform_fn`, which is a JAX function transforming
  a point from the reals into the support of the `log_density`. For example,
  if the `test_point` is a positive scalar, `transform_fn` may be `jnp.exp` or
  `jax.nn.softmax`.
  """
  _namespaces = None

  def _init(self):
    self._namespaces = []
    if is_tfp_bijector(self.transform_fn):
      self.transform_fn = self.transform_fn
      if self.inverse_transform_fn is None:
        self.inverse_transform_fn = self.transform_fn.inverse
      if self.inverse_log_det_jacobian is None:
        self.inverse_log_det_jacobian = (
            self.transform_fn.inverse_log_det_jacobian
        )
    if self.inverse_transform_fn is None:
      self.inverse_transform_fn = oryx.core.inverse(self.transform_fn)
    for module in _MODULES:
      k = _Namespace()
      module_name = module.__name__.split(".")[-1]
      for class_name in module.__all__:
        clas = getattr(module, class_name)
        k.__setclass__(clas, self)

      # guard against no optional libraries being installed.
      if module.__all__:
        setattr(self, module_name, k)
        self._namespaces.append(module_name)

  def __post_init__(self):
    self._init()
    super().__init__(log_density=self.log_density,
                     test_point=self.test_point,
                     transform_fn=self.transform_fn,
                     initial_state=self.initial_state,
                     inverse_transform_fn=self.inverse_transform_fn,
                     inverse_log_det_jacobian=self.inverse_log_det_jacobian)

  def __repr__(self):
    methods = []
    for name in self._namespaces:
      methods.append(name)
      k = getattr(self, name)
      methods.append("\t." + "\n\t.".join(str(k).split()))
    return "\n".join(methods)
