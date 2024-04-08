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

"""flowMC specific code."""
import arviz as az
from bayeux._src import shared
from flowMC import Sampler
from flowMC.nfmodel import realNVP
from flowMC.nfmodel import rqSpline
from flowMC.proposal import HMC
from flowMC.proposal import MALA
import jax
import jax.numpy as jnp


_NF_MODELS = {
    "real_nvp": realNVP.RealNVP,
    "masked_coupling_rq_spline": rqSpline.MaskedCouplingRQSpline,
}

_LOCAL_SAMPLERS = {"mala": MALA.MALA, "hmc": HMC.HMC}


def get_nf_model_kwargs(nf_model, n_features, kwargs):
  """Sets defaults and merges user-provided adaptation keywords."""
  defaults = {
      # RealNVP kwargs
      "n_hidden": 100,
      "n_layer": 10,
      # MaskedCouplingRQSpline kwargs
      "n_layers": 4,
      "num_bins": 8,
      "hidden_size": [64, 64],
      "spline_range": (-10.0, 10.0),
      "n_features": n_features,
  } | kwargs

  nf_model_kwargs, nf_model_required = shared.get_default_signature(
      nf_model)
  nf_model_kwargs.update(
      {k: defaults[k] for k in nf_model_required if k in defaults})
  nf_model_required.remove("key")
  nf_model_required.remove("kwargs")
  nf_model_required = nf_model_required - nf_model_kwargs.keys()

  if nf_model_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(nf_model_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  nf_model_kwargs.update(
      {k: defaults[k] for k in nf_model_kwargs if k in defaults})

  return nf_model_kwargs


def get_local_sampler_kwargs(local_sampler, log_density, n_features, kwargs):
  """Sets defaults and merges user-provided adaptation keywords."""

  defaults = {
      # HMC kwargs
      "condition_matrix": jnp.eye(n_features),
      "n_leapfrog": 10,
      # Both
      "step_size": 0.1,
      "logpdf": log_density
  } | kwargs

  sampler_kwargs, sampler_required = shared.get_default_signature(
      local_sampler)
  sampler_kwargs.setdefault("jit", True)
  sampler_kwargs.update(
      {k: defaults[k] for k in sampler_required if k in defaults})
  sampler_required = sampler_required - sampler_kwargs.keys()
  sampler_kwargs.update(
      {k: defaults[k] for k in sampler_kwargs if k in defaults})
  sampler_required = sampler_required - sampler_kwargs.keys()

  if sampler_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(sampler_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  return sampler_kwargs


def get_sampler_kwargs(sampler, n_features, kwargs):
  """Sets defaults and merges user-provided adaptation keywords."""
  #  We support `num_chains` everywhere else, so support it here.
  if "num_chains" in kwargs:
    kwargs["n_chains"] = kwargs["num_chains"]
  defaults = {
      "n_loop_training": 5,
      "n_loop_production": 5,
      "n_local_steps": 50,
      "n_global_steps": 50,
      "n_chains": 20,
      "n_epochs": 30,
      "learning_rate": 0.01,
      "max_samples": 10_000,
      "momentum": 0.9,
      "batch_size": 10_000,
      "use_global": True,
      "global_sampler": None,
      "logging": True,
      "keep_quantile": 0.,
      "local_autotune": None,
      "train_thinning": 1,
      "output_thinning": 1,
      "n_sample_max": 10_000,
      "precompile": False,
      "verbose": False,
      "n_dim": n_features,
      "data": {}} | kwargs
  sampler_kwargs, sampler_required = shared.get_default_signature(sampler)
  sampler_kwargs.update(
      {k: defaults[k] for k in sampler_required if k in defaults})
  sampler_required = (sampler_required -
                      {"nf_model", "local_sampler", "rng_key", "kwargs"})
  sampler_required = sampler_required - sampler_kwargs.keys()

  if sampler_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(sampler_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  return defaults | sampler_kwargs


class _FlowMCSampler(shared.Base):
  """Base class for flowmc samplers."""
  name: str = ""
  nf_model: str = ""
  local_sampler: str = ""

  def _get_aux(self):
    flat, unflatten = jax.flatten_util.ravel_pytree(self.test_point)

    @jax.vmap
    def flatten(pytree):
      return jax.flatten_util.ravel_pytree(pytree)[0]

    constrained_log_density = self.constrained_log_density()
    def log_density(x, _):
      return constrained_log_density(unflatten(x)).squeeze()

    return log_density, flatten, unflatten, flat.shape[0]

  def get_kwargs(self, **kwargs):
    nf_model = _NF_MODELS[self.nf_model]
    local_sampler = _LOCAL_SAMPLERS[self.local_sampler]
    log_density, flatten, unflatten, n_features = self._get_aux()

    nf_model_kwargs = get_nf_model_kwargs(nf_model, n_features, kwargs)
    local_sampler_kwargs = get_local_sampler_kwargs(
        local_sampler, log_density, n_features, kwargs)
    sampler = Sampler.Sampler
    sampler_kwargs = get_sampler_kwargs(sampler, n_features, kwargs)
    extra_parameters = {"flatten": flatten,
                        "unflatten": unflatten,
                        "num_chains": sampler_kwargs["n_chains"],
                        "return_pytree": kwargs.get("return_pytree", False)}

    return {nf_model: nf_model_kwargs,
            local_sampler: local_sampler_kwargs,
            sampler: sampler_kwargs,
            "extra_parameters": extra_parameters}

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    extra_parameters = kwargs["extra_parameters"]
    num_chains = extra_parameters["num_chains"]
    init_key, nf_key, seed = jax.random.split(seed, 3)
    initial_state = self.get_initial_state(
        init_key, num_chains=num_chains)
    initial_state = extra_parameters["flatten"](initial_state)
    nf_model = _NF_MODELS[self.nf_model]
    local_sampler = _LOCAL_SAMPLERS[self.local_sampler]

    model = nf_model(key=nf_key, **kwargs[nf_model])
    local_sampler = local_sampler(**kwargs[local_sampler])
    sampler = Sampler.Sampler
    nf_sampler = sampler(
        rng_key=seed,
        local_sampler=local_sampler,
        nf_model=model,
        **kwargs[sampler])
    nf_sampler.sample(initial_state, {})
    chains, *_ = nf_sampler.get_sampler_state().values()

    unflatten = jax.vmap(jax.vmap(extra_parameters["unflatten"]))
    pytree = self.transform_fn(unflatten(chains))
    if extra_parameters["return_pytree"]:
      return pytree
    else:
      if hasattr(pytree, "_asdict"):
        pytree = pytree._asdict()
      elif not isinstance(pytree, dict):
        pytree = {"var0": pytree}
      return az.from_dict(posterior=pytree)


class RealNVPMALA(_FlowMCSampler):
  name = "flowmc_realnvp_mala"
  nf_model = "real_nvp"
  local_sampler = "mala"


class RealNVPHMC(_FlowMCSampler):
  name = "flowmc_realnvp_hmc"
  nf_model = "real_nvp"
  local_sampler = "hmc"


class MaskedCouplingRQSplineMALA(_FlowMCSampler):
  name = "flowmc_rqspline_mala"
  nf_model = "masked_coupling_rq_spline"
  local_sampler = "mala"


class MaskedCouplingRQSplineHMC(_FlowMCSampler):
  name = "flowmc_rqspline_hmc"
  nf_model = "masked_coupling_rq_spline"
  local_sampler = "hmc"
