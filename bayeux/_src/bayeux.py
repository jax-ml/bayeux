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

"""API entry point."""

import dataclasses

from bayeux import mcmc
from bayeux import optimize
from bayeux import vi
from bayeux._src import shared
import jax
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
    self.methods = []

  def __repr__(self):
    return "\n".join(self.methods)

  def __setclass__(self, clas, parent):
    kwargs = {k: getattr(parent, k) for k in _REQUIRED_KWARGS}
    setattr(self, clas.name, clas(**kwargs))
    self.methods.append(clas.name)


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
    for key, values in self.methods.items():
      methods.append(key)
      methods.append("\t." + "\n\t.".join(values))
    return "\n".join(methods)

  @property
  def methods(self):
    methods = {}
    for name in self._namespaces:
      methods[name] = getattr(self, name).methods
    return methods

  @classmethod
  def from_tfp(cls, pinned_joint_distribution, initial_state=None):
    log_density = pinned_joint_distribution.log_prob
    test_point = pinned_joint_distribution.sample_unpinned(
        seed=jax.random.PRNGKey(0))
    transform_fn = (
        pinned_joint_distribution.experimental_default_event_space_bijector()
    )
    inverse_transform_fn = transform_fn.inverse
    inverse_log_det_jacobian = transform_fn.inverse_log_det_jacobian
    return cls(
        log_density=log_density,
        test_point=test_point,
        transform_fn=transform_fn,
        initial_state=initial_state,
        inverse_transform_fn=inverse_transform_fn,
        inverse_log_det_jacobian=inverse_log_det_jacobian)

  @classmethod
  def from_numpyro(cls, numpyro_fn, initial_state=None):
    import numpyro  # pylint: disable=g-import-not-at-top

    def log_density(*args, **kwargs):
      # This clause is only required because the tfp vi routine tries to
      # pass dictionaries as keyword arguments, so this allows either
      # log_density(params) or log_density(**params)
      if args:
        x = args[0]
      else:
        x = kwargs
      return numpyro.infer.util.log_density(numpyro_fn, (), {}, x)[0]

    test_point = numpyro.infer.Predictive(
        numpyro_fn, num_samples=1)(jax.random.PRNGKey(0))
    test_point = {k: v[0] for k, v in test_point.items() if k != "observed"}

    def transform_fn(x):
      return numpyro.infer.util.constrain_fn(numpyro_fn, (), {}, x)

    return cls(
        log_density=log_density,
        test_point=test_point,
        transform_fn=transform_fn,
        initial_state=initial_state)

  @classmethod
  def from_pymc(cls, pm_model, initial_state=None):
    import pymc as pm  # pylint: disable=g-import-not-at-top
    import pymc.sampling.jax as pm_jax  # pylint: disable=g-import-not-at-top

    class Inverse(pm.logprob.transforms.Transform):
      def __init__(self, transform):
        self._transform = transform

      def forward(self, value, *inputs):
        """Apply the transformation."""
        return self._transform.backward(value, *inputs)

      def backward(self, value, *inputs):
        return self._transform.forward(value, *inputs)

    uc_model = pm.model.transform.conditioning.remove_value_transforms(pm_model)
    logp = pm_jax.get_jaxified_logp(uc_model)

    rvs_to_inverse = {
        k: None if v is None else Inverse(v)
        for k, v in pm_model.rvs_to_transforms.items()}
    rvs = pm_model.free_RVs
    inv_rvs = pm.logprob.utils.replace_rvs_by_values(
        rvs,
        rvs_to_values=pm_model.rvs_to_values,
        rvs_to_transforms=rvs_to_inverse)
    values = pm_model.value_vars
    names = [v.name for v in rvs]

    fwd = pm_jax.get_jaxified_graph(
        inputs=values,
        outputs=pm_model.replace_rvs_by_values(rvs))

    bwd = pm_jax.get_jaxified_graph(
        inputs=values,
        outputs=pm_model.replace_rvs_by_values(inv_rvs))
    def logp_wrap(*args, **kwargs):
      # This clause is only required because the tfp vi routine tries to
      # pass dictionaries as keyword arguments, so this allows either
      # log_density(params) or log_density(**params)
      if args:
        kwargs = args[0]
      return logp([kwargs[k] for k in names])

    def fwd_wrap(args):
      ret = fwd(*[args[k] for k in names])
      return dict(zip(names, ret))

    def bwd_wrap(args):
      ret = bwd(*[args[k] for k in names])
      return dict(zip(names, ret))

    test_point = uc_model.initial_point()
    return cls(
        log_density=logp_wrap,
        test_point=test_point,
        transform_fn=fwd_wrap,
        inverse_transform_fn=bwd_wrap,
        initial_state=initial_state)
