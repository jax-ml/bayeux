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

"""NumPyro specific code."""

import arviz as az
from bayeux._src import shared
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp


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
