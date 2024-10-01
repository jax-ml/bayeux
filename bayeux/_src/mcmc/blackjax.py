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

"""BlackJAX specific code."""
import functools

import arviz as az
from bayeux._src import shared
import blackjax
import jax
import jax.numpy as jnp
import optax


def _get_run_inference_algorithm_kwarg_name():
  """This is a hack while blackjax changes API for run_inference_algorithm.

  This should be deleted and default to "initial_state" once the API
  stabilizes and we can depend on some version > 1.2.1.

  We do this out here so that it just runs once.

  Returns:
    keyword argument name that `blackjax.util.run_inference_algorithm` expects.
  """
  _, req = shared.get_default_signature(blackjax.util.run_inference_algorithm)
  if "initial_state_or_position" in req:
    return "initial_state_or_position"
  return "initial_state"
_INFERENCE_KWARG = _get_run_inference_algorithm_kwarg_name()


_ADAPT_FNS = {
    "window": blackjax.window_adaptation,
    "pathfinder": blackjax.pathfinder_adaptation,
    "chees": blackjax.chees_adaptation,
    "meads": blackjax.meads_adaptation,
}

_ALGORITHMS = {
    "hmc": blackjax.hmc,
    "ghmc": blackjax.ghmc,
    "dynamic_hmc": blackjax.dynamic_hmc,
    "nuts": blackjax.nuts,
}


def _convert_algorithm(algorithm):
  # Remove this after blackjax is stable
  if hasattr(algorithm, "differentiable"):
    return algorithm.differentiable
  return algorithm


def get_extra_kwargs(kwargs):
  defaults = {
      "chain_method": "vectorized",
      "num_chains": 8,
      "num_draws": 500,
      "num_adapt_draws": 500,
      "return_pytree": False}
  shared.update_with_kwargs(defaults, kwargs=kwargs)
  return defaults


class _BlackjaxSampler(shared.Base):
  """Base class for blackjax samplers."""
  name: str = ""
  adapt_fn: str = ""
  algorithm: str = ""

  def get_kwargs(self, **kwargs):
    adapt_fn = _ADAPT_FNS[self.adapt_fn]
    algorithm = _ALGORITHMS[self.algorithm]
    extra_parameters = get_extra_kwargs(kwargs)
    constrained_log_density = self.constrained_log_density()
    adaptation_kwargs, run_kwargs = get_adaptation_kwargs(
        adapt_fn, algorithm, constrained_log_density, extra_parameters | kwargs)
    return {adapt_fn: adaptation_kwargs,
            "adapt.run": run_kwargs,
            _convert_algorithm(algorithm): get_algorithm_kwargs(
                _convert_algorithm(algorithm), constrained_log_density, kwargs),
            "extra_parameters": extra_parameters}

  def __call__(self, seed, **kwargs):
    init_key, sample_key = jax.random.split(seed)
    kwargs = self.get_kwargs(**kwargs)
    initial_state = self.get_initial_state(
        init_key, num_chains=kwargs["extra_parameters"]["num_chains"])

    return _sample_blackjax(
        initial_state=self.inverse_transform_fn(initial_state),
        algorithm=_ALGORITHMS[self.algorithm],
        transform_fn=self.transform_fn,
        adapt_fn=_ADAPT_FNS[self.adapt_fn],
        seed=sample_key,
        kwargs=kwargs)


class _BlackjaxDynamicSampler(_BlackjaxSampler):
  """Base class for blackjax samplers."""

  def __call__(self, seed, **kwargs):
    init_key, sample_key = jax.random.split(seed)
    kwargs = self.get_kwargs(**kwargs)
    initial_state = self.get_initial_state(
        init_key, num_chains=kwargs["extra_parameters"]["num_chains"])

    return _sample_blackjax_dynamic(
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


class CheesHMC(_BlackjaxDynamicSampler):
  name = "blackjax_chees_hmc"
  adapt_fn = "chees"
  algorithm = "dynamic_hmc"


class MeadsHMC(_BlackjaxDynamicSampler):
  name = "blackjax_meads_hmc"
  adapt_fn = "meads"
  algorithm = "ghmc"


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


def _blackjax_adapt(
    seed,
    adapt_fn,
    kwarg_dict,
    **kwargs):
  adapt = adapt_fn(**kwarg_dict[adapt_fn])
  (last_state, parameters), _ = adapt.run(
      rng_key=seed, **kwargs,
      **kwarg_dict["adapt.run"])
  return last_state, parameters


def _blackjax_inference(
    seed,
    adapt_state,
    adapt_parameters,
    algorithm,
    num_draws,
    kwargs):
  """Run blackjax inference loop in a vmappable way.

  Args:
    seed: jax PRNGKey
    adapt_state: return value from a blackjax adaptation algorithm.
    adapt_parameters: return value from blackjax adaptation algorithm
    algorithm: name of algorithm to run.
    num_draws: number of iterations to run for.
    kwargs: Extra keyword arguments for algorithm.

  Returns:
  The results of the inference and the adaptation:
    (states, infos), adaptation_parameters
  """

  algorithm_kwargs = kwargs[_convert_algorithm(algorithm)] | adapt_parameters
  inference_algorithm = algorithm(**algorithm_kwargs)
  # This is protecting against a change in blackjax where the
  # return from `run_inference_algorithm` changes from
  # `_, states, infos` to `_, (states, infos)`. This one weird
  # trick handles both cases.
  ret = blackjax.util.run_inference_algorithm(
      rng_key=seed,
      inference_algorithm=inference_algorithm,
      num_steps=num_draws,
      progress_bar=False,
      **{_INFERENCE_KWARG: adapt_state})
  if len(ret) == 2:  # For newer blackjax versions (1.2.4+)
    return ret[1]
  else:  # Delete this once blackjax 1.2.4 is stable
    return ret[1:]


def _blackjax_inference_loop(
    seed,
    init_position,
    adapt_fn,
    algorithm,
    num_draws,
    kwargs):
  """Constructs and runs inference loop."""
  adapt_seed, inference_seed = jax.random.split(seed)
  adapt_state, adapt_parameters = _blackjax_adapt(
      adapt_seed, adapt_fn, kwarg_dict=kwargs, position=init_position)
  return _blackjax_inference(
      inference_seed,
      adapt_state,
      adapt_parameters,
      algorithm,
      num_draws,
      kwargs), adapt_parameters


def _blackjax_stats_to_dict(sample_stats, potential_energy, adapt_parameters):
  """Extract ArviZ compatible stats from blackjax sampler.

  Adapted from https://github.com/pymc-devs/pymc

  Args:
    sample_stats: Blackjax NUTSInfo object containing sampler statistics.
    potential_energy: Potential energy values of sampled positions.
    adapt_parameters: Parameters from adaptation.

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
  converted_stats = {"lp": potential_energy}
  step_size = adapt_parameters.get("step_size", None)
  if step_size is not None:
    if jnp.ndim(step_size) == 0:
      converted_stats["step_size"] = jnp.full_like(potential_energy, step_size)
    else:
      converted_stats["step_size"] = jnp.repeat(
          step_size[..., None], repeats=jnp.shape(potential_energy)[-1], axis=-1
      )
  for old_name, new_name in rename_key.items():
    value = getattr(sample_stats, old_name, None)
    if value is not None:
      converted_stats[new_name] = value
  return converted_stats


def get_adaptation_kwargs(adaptation_algorithm, algorithm, log_density, kwargs):
  """Sets defaults and merges user-provided adaptation keywords."""
  adaptation_kwargs, adaptation_required = shared.get_default_signature(
      adaptation_algorithm)
  adaptation_kwargs.update(
      {k: kwargs[k] for k in adaptation_required if k in kwargs})
  if "logdensity_fn" in adaptation_required:
    adaptation_kwargs["logdensity_fn"] = log_density
    adaptation_required.remove("logdensity_fn")
  elif "logprob_fn" in adaptation_required:
    adaptation_kwargs["logprob_fn"] = log_density
    adaptation_required.remove("logprob_fn")

  adaptation_required.discard("extra_parameters")
  if "algorithm" in adaptation_required:
    adaptation_required.remove("algorithm")
    adaptation_kwargs["algorithm"] = algorithm
    adaptation_kwargs = (
        get_algorithm_kwargs(_convert_algorithm(algorithm), log_density, kwargs)
        | adaptation_kwargs)

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
  adaptation_kwargs.pop("step_size", None)
  # blackjax doesn't have a pleasant way to accept this argument --
  # window_adaptation calls `algorithm.build_kernel()` with no arguments, but
  # it should probably take the below arguments:
  adaptation_kwargs.pop("divergence_threshold", None)
  adaptation_kwargs.pop("integrator", None)
  adaptation_kwargs.pop("max_num_doublings", None)

  adapt = adaptation_algorithm(**adaptation_kwargs)
  run_kwargs, run_required = shared.get_default_signature(adapt.run)
  run_required.remove("rng_key")
  run_kwargs.update({k: kwargs[k] for k in run_required if k in kwargs})
  if "optim" in run_required:
    run_kwargs["optim"] = optax.adam(learning_rate=0.01)
    run_required.remove("optim")
  if "step_size" in run_required:
    run_kwargs["step_size"] = 0.5
    run_required.remove("step_size")
  run_kwargs["num_steps"] = kwargs.get("num_adapt_draws",
                                       run_kwargs["num_steps"])

  return adaptation_kwargs, run_kwargs


def get_algorithm_kwargs(algorithm, log_density, kwargs):
  """Sets defaults and merges user-provided keywords for sampling."""
  algorithm_kwargs, algorithm_required = shared.get_default_signature(algorithm)
  kwargs_with_defaults = {
      "logdensity_fn": log_density,
      "step_size": 0.5,
      "num_integration_steps": 16,
  } | kwargs
  shared.update_with_kwargs(
      algorithm_kwargs, reqd=algorithm_required, kwargs=kwargs_with_defaults)
  algorithm_required.remove("logdensity_fn")
  algorithm_required.discard("inverse_mass_matrix")
  algorithm_required.discard("alpha")
  algorithm_required.discard("delta")
  algorithm_required.discard("momentum_inverse_scale")

  algorithm_required = algorithm_required - algorithm_kwargs.keys()
  if algorithm_required:
    raise ValueError(f"Unexpected required arguments: "
                     f"{','.join(algorithm_required)}. Probably file a bug, but"
                     " you can try to manually supply them as keywords.")
  return algorithm_kwargs


def _sample_blackjax_dynamic(
    *,
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

  adapt_seed, seed = jax.random.split(seed)
  adapt_state, adapt_parameters = _blackjax_adapt(
      seed=adapt_seed,
      adapt_fn=adapt_fn,
      kwarg_dict=kwargs,
      positions=initial_state,
  )
  sampler = functools.partial(
      _blackjax_inference,
      adapt_parameters=adapt_parameters,
      algorithm=algorithm,
      num_draws=num_draws,
      kwargs=kwargs)
  map_seed = jax.random.split(seed, num_chains)
  mapped_sampler = shared.map_fn(chain_method, sampler)

  states, stats = mapped_sampler(map_seed, adapt_state)
  draws = transform_fn(states.position)
  if extra_parameters["return_pytree"]:
    return draws
  else:
    potential_energy = states.logdensity
    sample_stats = _blackjax_stats_to_dict(
        stats, potential_energy, adapt_parameters)
    if hasattr(draws, "_asdict"):
      draws = draws._asdict()
    elif not isinstance(draws, dict):
      draws = {"var0": draws}
    return az.from_dict(posterior=draws, sample_stats=sample_stats)


def _sample_blackjax(
    *,
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
  sampler = functools.partial(
      _blackjax_inference_loop,
      algorithm=algorithm,
      adapt_fn=adapt_fn,
      num_draws=num_draws,
      kwargs=kwargs)
  map_seed = jax.random.split(seed, num_chains)
  mapped_sampler = shared.map_fn(chain_method, sampler)

  (states, stats), adapt_parameters = mapped_sampler(map_seed, initial_state)
  draws = transform_fn(states.position)
  if extra_parameters["return_pytree"]:
    return draws
  else:
    potential_energy = states.logdensity
    sample_stats = _blackjax_stats_to_dict(
        stats, potential_energy, adapt_parameters)
    if hasattr(draws, "_asdict"):
      draws = draws._asdict()
    elif not isinstance(draws, dict):
      draws = {"var0": draws}
    return az.from_dict(posterior=draws, sample_stats=sample_stats)
