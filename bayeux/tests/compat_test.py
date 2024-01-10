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

"""Tests for bayeux.Model working with other libraries."""
import bayeux as bx
import jax
import numpy as np
import numpyro
import tensorflow_probability.substrates.jax as tfp

dist = numpyro.distributions
tfd = tfp.distributions


def test_from_numpyro():
  treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
  treatment_stddevs = np.array(
      [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)

  def numpyro_model():
    avg_effect = numpyro.sample('avg_effect', dist.Normal(0.0, 10.0))
    avg_stddev = numpyro.sample('avg_stddev', dist.HalfNormal(10.0))
    with numpyro.plate('J', 8):
      school_effects = numpyro.sample('school_effects', dist.Normal(0.0, 1.0))
      numpyro.sample(
          'observed',
          dist.Normal(
              avg_effect[..., None] + avg_stddev[..., None] * school_effects,
              treatment_stddevs),
          obs=treatment_effects)

  bx_model = bx.Model.from_numpyro(numpyro_model)
  ret = bx_model.optimize.optax_adam(seed=jax.random.key(0))
  assert ret is not None


def test_from_tfp():
  treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
  treatment_stddevs = np.array(
      [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)

  @tfd.JointDistributionCoroutineAutoBatched
  def tfp_model():
    avg_effect = yield tfd.Normal(0., 10., name='avg_effect')
    avg_stddev = yield tfd.HalfNormal(10., name='avg_stddev')

    school_effects = yield tfd.Sample(
        tfd.Normal(0., 1.), sample_shape=8, name='school_effects')
    yield tfd.Normal(
        avg_effect[..., None] + avg_stddev[..., None] * school_effects,
        treatment_stddevs, name='observed')

  pinned_model = tfp_model.experimental_pin(observed=treatment_effects)
  bx_model = bx.Model.from_tfp(pinned_model)
  ret = bx_model.optimize.optax_adam(seed=jax.random.key(0))
  assert ret is not None
