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

"""Nutpie specific code."""

import arviz as az
from bayeux._src import shared
import jax
import numpy as np

import nutpie
from nutpie.compiled_pyfunc import from_pyfunc


class _NutpieSampler(shared.Base):
  """Base class for nutpie sampler."""
  name: str = "nutpie"

  def _get_aux(self):
    flat, unflatten = jax.flatten_util.ravel_pytree(self.test_point)

    def flatten(pytree):
      return jax.flatten_util.ravel_pytree(pytree)[0]

    def make_logp_fn():
      constrained_log_density = self.constrained_log_density()
      def log_density(x):
        return constrained_log_density(unflatten(x)).squeeze()
      log_grad = jax.jit(jax.value_and_grad(log_density))
      def wrapper(x):
        val, grad = log_grad(x)
        return val, np.array(grad, dtype=np.float64)
      return wrapper
    return make_logp_fn, flatten, unflatten, flat.shape[0]

  def get_kwargs(self, **kwargs):
    make_logp_fn, flatten, unflatten, ndim = self._get_aux()

    def make_expand_fn(*args, **kwargs):
      del args
      del kwargs
      return lambda x: {"x": np.asarray(x, dtype="float64")}

    from_pyfunc_kwargs = {
        "ndim": ndim,
        "make_logp_fn": make_logp_fn,
        "make_expand_fn": make_expand_fn,
        "expanded_shapes": [(ndim,)],
        "expanded_names": ["x"],
        "expanded_dtypes": [np.float64],
    }
    from_pyfunc_kwargs = {
        k: kwargs.get(k, v) for k, v in from_pyfunc_kwargs.items()}

    kwargs_with_defaults = {
        "draws": 1_000,
        "chains": 8,
    } | kwargs
    sample_kwargs, _ = shared.get_default_signature(nutpie.sample)
    sample_kwargs.update({k: kwargs_with_defaults[k] for k in sample_kwargs if
                          k in kwargs_with_defaults})
    if "cores" not in kwargs:
      sample_kwargs["cores"] = sample_kwargs["chains"]
    extra_parameters = {"flatten": flatten,
                        "unflatten": unflatten,
                        "return_pytree": kwargs.get("return_pytree", False)}

    return {from_pyfunc: from_pyfunc_kwargs,
            nutpie.sample: sample_kwargs,
            "extra_parameters": extra_parameters}

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    extra_parameters = kwargs["extra_parameters"]
    compiled = from_pyfunc(**kwargs[from_pyfunc])
    idata = nutpie.sample(compiled_model=compiled,
                          **kwargs[nutpie.sample])
    return _postprocess_idata(idata,
                              extra_parameters["unflatten"],
                              self.transform_fn,
                              extra_parameters["return_pytree"])


def _pytree_to_dict(draws):
  if hasattr(draws, "_asdict"):
    draws = draws._asdict()
  elif not isinstance(draws, dict):
    draws = {"var0": draws}

  return draws


def _postprocess_idata(idata, unflatten, transform_fn, return_pytree):
  """Convert nutpie inference data back to pytree, transform, and put back."""
  unflatten = jax.vmap(jax.vmap(unflatten))
  posterior = transform_fn(unflatten(idata.posterior.x.values))

  if return_pytree:
    return posterior

  posterior = _pytree_to_dict(posterior)
  warmup_posterior = _pytree_to_dict(
      transform_fn(unflatten(idata.warmup_posterior.x.values)))
  new_posterior = az.from_dict(posterior=posterior)
  new_warmup_posterior = az.from_dict(posterior=warmup_posterior)
  del idata.posterior
  del idata.warmup_posterior
  idata.add_groups(posterior=new_posterior.posterior,
                   warmup_posterior=new_warmup_posterior.posterior)
  return idata
