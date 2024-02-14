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

"""TFP specific code."""

import arviz as az
from bayeux._src import shared
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp


_ALGORITHMS = {
    "hmc": tfp.mcmc.HamiltonianMonteCarlo,
    "nuts": tfp.mcmc.NoUTurnSampler,
}

_TRACE_FNS = {
    "hmc": tfp.experimental.mcmc.windowed_sampling.default_hmc_trace_fn,
    "nuts": tfp.experimental.mcmc.windowed_sampling.default_nuts_trace_fn,
}


class SnaperHMC(shared.Base):
  """Implements SNAPER HMC [1] with step size adaptation.

  [1]: Sountsov, P. & Hoffman, M. (2021). Focusing on Difficult Directions for
  Learning HMC Trajectory Lengths. <https://arxiv.org/abs/2110.11576>
  """
  name = "tfp_snaper_hmc"

  def get_kwargs(self, **kwargs):
    kwargs_with_defaults = {
        "num_results": 1_000,
        "num_chains": 8,
    } | kwargs
    snaper = tfp.experimental.mcmc.sample_snaper_hmc
    snaper_kwargs, snaper_required = shared.get_default_signature(snaper)
    snaper_kwargs.update({k: kwargs_with_defaults[k] for k in snaper_required
                          if k in kwargs_with_defaults})
    snaper_required.remove("model")
    # Initial state is handled internally
    snaper_kwargs.pop("init_state")
    # Seed set later
    snaper_kwargs.pop("seed")

    snaper_required = snaper_required - snaper_kwargs.keys()

    if snaper_required:
      raise ValueError(f"Unexpected required arguments: "
                       f"{','.join(snaper_required)}. Probably file a bug, but "
                       "you can try to manually supply them as keywords.")
    snaper_kwargs.update({k: kwargs_with_defaults[k] for k in snaper_kwargs
                          if k in kwargs_with_defaults})
    return {
        snaper: snaper_kwargs,
        "extra_parameters": {
            "return_pytree": kwargs.get("return_pytree", False)
        },
    }

  def __call__(self, seed, **kwargs):
    snaper = tfp.experimental.mcmc.sample_snaper_hmc
    init_key, sample_key = jax.random.split(seed)
    kwargs = self.get_kwargs(**kwargs)
    initial_state = self.get_initial_state(
        init_key, num_chains=kwargs[snaper]["num_chains"])

    vmapped_constrained_log_prob = jax.vmap(self.constrained_log_density())

    def tlp(*args, **kwargs):
      if args:
        return vmapped_constrained_log_prob(args)
      else:
        return vmapped_constrained_log_prob(kwargs)

    (draws, trace), *_ = snaper(
        model=tlp, init_state=initial_state, seed=sample_key, **kwargs[snaper]
    )
    draws = self.transform_fn(draws)
    if kwargs["extra_parameters"]["return_pytree"]:
      return draws

    if hasattr(draws, "_asdict"):
      draws = draws._asdict()
    elif not isinstance(draws, dict):
      draws = {"var0": draws}

    draws = {x: np.swapaxes(v, 0, 1) for x, v in draws.items()}
    return az.from_dict(posterior=draws, sample_stats=_tfp_stats_to_dict(trace))


class _TFPBase(shared.Base):
  """Base class for TFP windowed samplers."""
  name: str = ""
  algorithm: str = ""

  def get_kwargs(self, **kwargs):
    if self.algorithm == "nuts":
      target_accept_prob = 0.8
    else:
      target_accept_prob = 0.6

    kwargs = {
        "target_accept_prob": target_accept_prob,
        "reduce_fn": tfp.math.reduce_log_harmonic_mean_exp,
        "num_adaptation_steps": 500,
        "step_size": 0.5,
        "num_leapfrog_steps": 8,
    } | kwargs
    extra_parameters = {
        "num_draws": 1_000,
        "num_chains": 8,
        "num_adaptation_steps": 500,
        "return_pytree": False,
    }
    shared.update_with_kwargs(extra_parameters, kwargs=kwargs)

    dual_averaging_kwargs, da_reqd = shared.get_default_signature(
        tfp.mcmc.DualAveragingStepSizeAdaptation)
    shared.update_with_kwargs(
        dual_averaging_kwargs, reqd=da_reqd, kwargs=kwargs)
    da_reqd = da_reqd - dual_averaging_kwargs.keys()
    da_reqd.remove("inner_kernel")
    if da_reqd:
      raise ValueError(
          "Unexpected required arguments: "
          f"{','.join(da_reqd)}. Probably file a bug, but "
          "you can try to manually supply them as keywords."
      )

    proposal_kwargs, proposal_reqd = shared.get_default_signature(
        _ALGORITHMS[self.algorithm])
    shared.update_with_kwargs(
        proposal_kwargs, reqd=proposal_reqd, kwargs=kwargs)
    proposal_reqd = proposal_reqd - proposal_kwargs.keys()
    proposal_reqd.remove("target_log_prob_fn")
    if proposal_reqd:
      raise ValueError(
          "Unexpected required arguments: "
          f"{','.join(proposal_reqd)}. Probably file a bug, but "
          "you can try to manually supply them as keywords."
      )
    return {
        "extra_parameters": extra_parameters,
        "dual_averaging_kwargs": dual_averaging_kwargs,
        "proposal_kernel_kwargs": proposal_kwargs}

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    init_key, sample_key = jax.random.split(seed)

    extra_parameters = kwargs["extra_parameters"]
    dual_averaging_kwargs = kwargs["dual_averaging_kwargs"]
    proposal_kernel_kwargs = kwargs["proposal_kernel_kwargs"]
    initial_state = self.get_initial_state(
        init_key, num_chains=extra_parameters["num_chains"])

    vmapped_constrained_log_density = jax.vmap(self.constrained_log_density())
    initial_transformed_position, treedef = jax.tree_util.tree_flatten(
        self.inverse_transform_fn(initial_state))

    def target_log_prob_fn(*args):
      return vmapped_constrained_log_density(
          jax.tree_util.tree_unflatten(treedef, args))
    proposal_kernel_kwargs["target_log_prob_fn"] = target_log_prob_fn

    initial_running_variance = [
        tfp.experimental.stats.sample_stats.RunningVariance.from_stats(
            num_samples=jnp.array(1, part.dtype),
            mean=jnp.zeros_like(part),
            variance=jnp.ones_like(part))
        for part in initial_transformed_position]

    # The public API expects a JointDistribution. Much of the above is adapted
    # from the source code for
    # `tfp.experimental.mcmc.windowed_adaptive_{nuts|hmc}`, but handling a raw
    # log density, and doing the structure flattening with `jax.tree_utils`.
    draws, trace = tfp.experimental.mcmc.windowed_sampling._do_sampling(
        kind=self.algorithm,
        proposal_kernel_kwargs=proposal_kernel_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        num_draws=extra_parameters["num_draws"],
        num_burnin_steps=extra_parameters["num_adaptation_steps"],
        initial_position=initial_transformed_position,
        initial_running_variance=initial_running_variance,
        bijector=None,
        trace_fn=_TRACE_FNS[self.algorithm],
        return_final_kernel_results=False,
        chain_axis_names=None,
        shard_axis_names=None,
        seed=sample_key)

    draws = self.transform_fn(jax.tree_util.tree_unflatten(treedef, draws))
    if extra_parameters["return_pytree"]:
      return draws

    if hasattr(draws, "_asdict"):
      draws = draws._asdict()
    elif not isinstance(draws, dict):
      draws = {"var0": draws}

    draws = {x: np.swapaxes(v, 0, 1) for x, v in draws.items()}
    return az.from_dict(posterior=draws, sample_stats=_tfp_stats_to_dict(trace))


class NUTS(_TFPBase):
  name = "tfp_nuts"
  algorithm = "nuts"


class HMC(_TFPBase):
  name = "tfp_hmc"
  algorithm = "hmc"


def _tfp_stats_to_dict(stats):
  new_stats = {}
  for k, v in stats.items():
    if k == "variance_scaling":
      continue
    if np.ndim(v) > 1:
      new_stats[k] = np.swapaxes(v, 0, 1)
    else:
      new_stats[k] = v
  return new_stats
