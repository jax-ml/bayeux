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

"""NumPyro specific code."""

import arviz as az
from bayeux._src import shared
import jax
import numpyro.infer


def _potential(log_density):

  def mapped(*args, **kwargs):
    return -log_density(*args, **kwargs)

  return mapped

_ALGORITHMS = {"hmc": numpyro.infer.HMC, "nuts": numpyro.infer.NUTS}


class _NumpyroSampler(shared.Base):
  """Base class for NumPyro samplers."""
  name: str = ""
  algorithm: str = ""

  def get_kwargs(self, **kwargs):
    algorithm = _ALGORITHMS[self.algorithm]
    return {
        algorithm: get_sampler_kwargs(algorithm, kwargs),
        numpyro.infer.MCMC: get_mcmc_kwargs(kwargs),
        "extra_parameters": {
            "return_pytree": kwargs.get("return_pytree", False)
        },
    }

  def __call__(self, seed, **kwargs):
    init_key, sample_key = jax.random.split(seed)
    kwargs = self.get_kwargs(**kwargs)
    initial_state = self.get_initial_state(
        init_key, num_chains=kwargs[numpyro.infer.MCMC]["num_chains"])

    potential_fn = _potential(self.constrained_log_density())
    return _sample_numpyro(
        potential_fn=potential_fn,
        initial_state=initial_state,
        transform_fn=self.transform_fn,
        inverse_transform_fn=self.inverse_transform_fn,
        algorithm=_ALGORITHMS[self.algorithm],
        seed=sample_key,
        kwargs=kwargs)


class HMC(_NumpyroSampler):
  name = "numpyro_hmc"
  algorithm = "hmc"


class NUTS(_NumpyroSampler):
  name = "numpyro_nuts"
  algorithm = "nuts"


def get_sampler_kwargs(algorithm, kwargs):
  """Construct default args and include user arguments for samplers."""
  sampler_kwargs, sampler_required = shared.get_default_signature(algorithm)
  shared.update_with_kwargs(
      sampler_kwargs, reqd=sampler_required, kwargs=kwargs)
  sampler_kwargs.pop("potential_fn")

  sampler_required = sampler_required - sampler_kwargs.keys()

  if sampler_required:
    raise ValueError(f"Unexpected required arguments: "
                     f"{','.join(sampler_required)}. Probably file a bug, but "
                     "you can try to manually supply them as keywords.")
  return sampler_kwargs


def get_mcmc_kwargs(kwargs):
  """Construct default args and include user arguments for MCMC."""
  mcmc_kwargs, mcmc_required = shared.get_default_signature(numpyro.infer.MCMC)
  kwargs_with_defaults = {
      "num_warmup": 500,
      "num_samples": 1000,
      "num_chains": 8,
      "chain_method": "vectorized",
  } | kwargs
  shared.update_with_kwargs(
      mcmc_kwargs, reqd=mcmc_required, kwargs=kwargs_with_defaults)
  mcmc_required = mcmc_required - mcmc_kwargs.keys()
  mcmc_required.remove("sampler")
  if mcmc_required:
    raise ValueError(f"Unexpected required arguments: "
                     f"{','.join(mcmc_required)}. Probably file a bug, but "
                     "you can try to manually supply them as keywords.")
  return mcmc_kwargs


def _sample_numpyro(*, potential_fn, initial_state, transform_fn,
                    inverse_transform_fn, algorithm, seed, kwargs):
  """Run MCMC using NumPyro."""
  kernel = algorithm(
      potential_fn=potential_fn,
      **kwargs[algorithm],
  )

  mcmc_kernel = numpyro.infer.MCMC(
      sampler=kernel,
      **kwargs[numpyro.infer.MCMC],
  )

  init_params = inverse_transform_fn(initial_state)
  mcmc_kernel.run(
      seed,
      init_params=init_params,
      extra_fields=(
          "num_steps",
          "potential_energy",
          "energy",
          "adapt_state.step_size",
          "accept_prob",
          "diverging",
      ),
  )
  samples = mcmc_kernel.get_samples(group_by_chain=True)
  if kwargs["extra_parameters"]["return_pytree"]:
    return transform_fn(samples)
  else:
    mcmc_kernel.get_samples = lambda group_by_chain=True: transform_fn(samples)
    return az.from_numpyro(mcmc_kernel)
