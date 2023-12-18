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

"""BlackJAX specific code."""
import functools

import arviz as az
from bayeux._src import shared
import blackjax
import jax


_ADAPT_FNS = {
    "window": blackjax.window_adaptation,
    "pathfinder": blackjax.pathfinder_adaptation,
}

_ALGORITHMS = {
    "hmc": blackjax.hmc,
    "nuts": blackjax.nuts,
}


def get_extra_kwargs(kwargs):
  return {
      "chain_method": "vectorized",
      "num_chains": 8,
      "num_draws": 500,
      "num_adapt_draws": 500,
      "return_pytree": False,
  } | kwargs


class _BlackjaxSampler(shared.Base):
  """Base class for blackjax samplers."""
  name: str = ""
  adapt_fn: str = ""
  algorithm: str = ""

  def get_kwargs(self, **kwargs):
    adapt_fn = _ADAPT_FNS[self.adapt_fn]
    algorithm = _ALGORITHMS[self.algorithm]
    return {adapt_fn: get_adaptation_kwargs(adapt_fn, algorithm, kwargs),
            algorithm: get_algorithm_kwargs(algorithm, kwargs),
            "extra_parameters": get_extra_kwargs(kwargs)}

  def __call__(self, seed, **kwargs):
    init_key, sample_key = jax.random.split(seed)
    kwargs = self.get_kwargs(**kwargs)
    initial_state = self.get_initial_state(
        init_key, num_chains=kwargs["extra_parameters"]["num_chains"])

    return _sample_blackjax(
        log_density=self.constrained_log_density(),
        initial_state=self.inverse_transform_fn(initial_state),
        algorithm=_ALGORITHMS[self.algorithm],
        transform_fn=self.transform_fn,
        adapt_fn=_ADAPT_FNS[self.adapt_fn],
        seed=sample_key,
        kwargs=kwargs)


class HMC(_BlackjaxSampler):
  name = "blackjax_hmc"
  adapt_fn = "window"
  algorithm = "hmc"


class HMCPathfinder(_BlackjaxSampler):
  name = "blackjax_hmc_pathfinder"
  adapt_fn = "pathfinder"
  algorithm = "hmc"


class NUTS(_BlackjaxSampler):
  name = "blackjax_nuts"
  adapt_fn = "window"
  algorithm = "nuts"


class NUTSPathfinder(_BlackjaxSampler):
  name = "blackjax_nuts_pathfinder"
  adapt_fn = "pathfinder"
  algorithm = "nuts"


def _blackjax_inference_loop(
    seed,
    init_position,
    adapt_fn,
    algorithm,
    log_density,
    num_draws,
    num_adapt_draws,
    kwargs):
  """Constructs and runs inference loop."""
  adapt_seed, inference_seed = jax.random.split(seed)
  adapt = adapt_fn(logdensity_fn=log_density, **kwargs[adapt_fn])
  (last_state, parameters), _ = adapt.run(
      rng_key=adapt_seed, position=init_position, num_steps=num_adapt_draws)

  algorithm_kwargs = kwargs[algorithm] | parameters
  kernel = algorithm(log_density, **algorithm_kwargs).step

  @jax.jit
  def inference_loop(rng_key):

    def one_step(state, rng_key):
      state, info = kernel(rng_key, state)
      return state, (state, info)

    keys = jax.random.split(rng_key, num_draws)
    _, (states, infos) = jax.lax.scan(one_step, last_state, keys)

    return states, infos

  return inference_loop(inference_seed)


def _blackjax_stats_to_dict(sample_stats, potential_energy):
  """Extract ArviZ compatible stats from blackjax sampler.

  Adapted from https://github.com/pymc-devs/pymc

  Args:
    sample_stats: Blackjax NUTSInfo object containing sampler statistics.
    potential_energy: Potential energy values of sampled positions.

  Returns:
      Dictionary of sampler statistics.
  """
  rename_key = {
      "is_divergent": "diverging",
      "energy": "energy",
      "num_trajectory_expansions": "tree_depth",
      "num_integration_steps": "n_steps",
      "acceptance_rate": "acceptance_rate",         # naming here depends
      "acceptance_probability": "acceptance_rate",  # on blackjax version
  }
  converted_stats = {}
  converted_stats["lp"] = potential_energy
  for old_name, new_name in rename_key.items():
    value = getattr(sample_stats, old_name, None)
    if value is not None:
      converted_stats[new_name] = value
  return converted_stats


def get_adaptation_kwargs(adaptation_algorithm, algorithm, kwargs):
  """Sets defaults and merges user-provided adaptation keywords."""
  adaptation_kwargs, adaptation_required = shared.get_default_signature(
      adaptation_algorithm)
  adaptation_kwargs.update(
      {k: kwargs[k] for k in adaptation_required if k in kwargs})
  adaptation_required.remove("logdensity_fn")
  adaptation_required.remove("extra_parameters")
  adaptation_required.remove("algorithm")
  adaptation_kwargs["algorithm"] = algorithm
  adaptation_kwargs = (
      get_algorithm_kwargs(algorithm, kwargs) | adaptation_kwargs)

  adaptation_required = adaptation_required - adaptation_kwargs.keys()

  if adaptation_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(adaptation_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  adaptation_kwargs.update(
      {k: kwargs[k] for k in adaptation_kwargs if k in kwargs}
  )
  # step_size will get adapted -- maybe warn if this is set manually, and
  # suggest setting init_step_size instead?
  adaptation_kwargs.pop("step_size")
  # blackjax doesn't have a pleasant way to accept this argument --
  # window_adaptation calls `algorithm.build_kernel()` with no arguments, but
  # it should probably take the below arguments:
  adaptation_kwargs.pop("divergence_threshold", None)
  adaptation_kwargs.pop("integrator", None)
  adaptation_kwargs.pop("max_num_doublings", None)

  return adaptation_kwargs


def get_algorithm_kwargs(algorithm, kwargs):
  """Sets defaults and merges user-provided keywords for sampling."""
  algorithm_kwargs, algorithm_required = shared.get_default_signature(algorithm)
  kwargs_with_defaults = {
      "step_size": 0.01,
      "num_integration_steps": 8,
  } | kwargs
  algorithm_kwargs.update(
      {
          k: kwargs_with_defaults[k]
          for k in algorithm_required
          if k in kwargs_with_defaults
      })
  algorithm_required.remove("logdensity_fn")
  algorithm_required.remove("inverse_mass_matrix")

  algorithm_required = algorithm_required - algorithm_kwargs.keys()
  if algorithm_required:
    raise ValueError(f"Unexpected required arguments: "
                     f"{','.join(algorithm_required)}. Probably file a bug, but"
                     " you can try to manually supply them as keywords.")
  algorithm_kwargs.update(
      {
          k: kwargs_with_defaults[k]
          for k in algorithm_kwargs
          if k in kwargs_with_defaults})
  return algorithm_kwargs


def _sample_blackjax(
    *,
    log_density,
    initial_state,
    algorithm,
    seed,
    transform_fn,
    adapt_fn,
    kwargs):
  """Constructs and runs blackjax sampler."""
  extra_parameters = kwargs.pop("extra_parameters")
  num_draws = extra_parameters["num_draws"]
  num_chains = extra_parameters["num_chains"]
  chain_method = extra_parameters["chain_method"]
  num_adapt_draws = extra_parameters["num_adapt_draws"]
  sampler = functools.partial(
      _blackjax_inference_loop,
      log_density=log_density,
      algorithm=algorithm,
      adapt_fn=adapt_fn,
      num_draws=num_draws,
      num_adapt_draws=num_adapt_draws,
      kwargs=kwargs)
  map_seed = jax.random.split(seed, num_chains)
  if chain_method == "parallel":
    mapped_sampler = jax.pmap(sampler)
  elif chain_method == "vectorized":
    mapped_sampler = jax.vmap(sampler)
  elif chain_method == "sequential":
    mapped_sampler = functools.partial(jax.tree_map, sampler)
  else:
    raise ValueError(f"Chain method {chain_method} not supported.")

  states, stats = mapped_sampler(map_seed, initial_state)
  draws = transform_fn(states.position)
  if extra_parameters["return_pytree"]:
    return draws
  else:
    potential_energy = states.logdensity
    sample_stats = _blackjax_stats_to_dict(stats, potential_energy)
    if hasattr(draws, "_asdict"):
      draws = draws._asdict()
    elif not isinstance(draws, dict):
      draws = {"var0": draws}
    return az.from_dict(posterior=draws, sample_stats=sample_stats)
