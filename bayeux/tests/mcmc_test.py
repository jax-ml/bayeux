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

"""Tests for the API."""
import importlib

import arviz as az
import bayeux as bx
import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp

jax.config.update("jax_threefry_partitionable", False)

tfd = tfp.distributions

METHODS = [getattr(bx.mcmc, k).name for k in bx.mcmc.__all__]


def max_rhat(idata):
  """Ugh."""
  return az.rhat(idata).max().to_array().to_numpy().max()


def test_methods_exist():
  # Can increment this later, but only need it to fail if methods disappear.
  assert len(METHODS) >= 6


def test_return_pytree_blackjax():
  model = bx.Model(log_density=lambda pt: -pt["x"]["y"]**2,
                   test_point={"x": {"y": jnp.array(1.)}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.blackjax_hmc(
      seed=seed,
      return_pytree=True,
      num_chains=4,
      num_draws=10,
      num_adapt_draws=10,
  )
  assert pytree["x"]["y"].shape == (4, 10)


def test_return_pytree_numpyro():
  model = bx.Model(log_density=lambda pt: -pt["x"]["y"]**2,
                   test_point={"x": {"y": jnp.array(1.)}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.numpyro_hmc(
      seed=seed,
      return_pytree=True,
      num_chains=4,
      num_samples=10,
      num_warmup=10,
  )
  assert pytree["x"]["y"].shape == (4, 10)


def test_return_pytree_tfp():
  model = bx.Model(log_density=lambda pt: -pt["x"]["y"]**2,
                   test_point={"x": {"y": jnp.array(1.)}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.tfp_snaper_hmc(
      seed=seed,
      return_pytree=True,
      num_chains=4,
      num_results=10,
      num_burnin_steps=10,
  )
  assert pytree["x"]["y"].shape == (10, 4)


def test_return_pytree_tfp_nuts():
  model = bx.Model(log_density=lambda pt: -pt["x"]["y"]**2,
                   test_point={"x": {"y": jnp.array(1.)}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.tfp_nuts(
      seed=seed,
      return_pytree=True,
      num_chains=4,
      num_draws=10,
      num_adaptation_steps=10,
  )
  assert pytree["x"]["y"].shape == (10, 4)


@pytest.mark.skipif(importlib.util.find_spec("flowMC") is None,
                    reason="Test requires flowMC which is not installed")
def test_return_pytree_flowmc():
  model = bx.Model(log_density=lambda pt: -jnp.sum(pt["x"]["y"]**2),
                   test_point={"x": {"y": jnp.array([1., 1.])}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.flowmc_realnvp_mala(
      seed=seed,
      return_pytree=True,
      n_chains=4,
      n_local_steps=1,
      n_global_steps=1,
      n_loop_training=1,
      n_loop_production=5,
  )
  # 10 draws = (1 local + 1 global) * 5 loops
  assert pytree["x"]["y"].shape == (4, 10, 2)


@pytest.mark.skipif(importlib.util.find_spec("nutpie") is None,
                    reason="Test requires nutpie which is not installed")
def test_return_pytree_nutpie():
  model = bx.Model(log_density=lambda pt: -jnp.sum(pt["x"]["y"]**2),
                   test_point={"x": {"y": jnp.array([1., 1.])}})
  seed = jax.random.PRNGKey(0)
  pytree = model.mcmc.nutpie(
      seed=seed,
      return_pytree=True,
      chains=4,
      draws=10,
      tune=10,
  )
  # 10 draws = (1 local + 1 global) * 5 loops
  assert pytree["x"]["y"].shape == (4, 10, 2)


@pytest.mark.parametrize("method", METHODS)
def test_samplers(method):
  # flowMC samplers are broken for 0 or 1 dimensions, so just test
  # everything on 2 dimensions for now.
  model = bx.Model(log_density=lambda pt: -pt["x"]**2,
                   test_point={"x": 1.})
  sampler = getattr(model.mcmc, method)
  seed = jax.random.PRNGKey(0)
  assert sampler.debug(seed=seed, verbosity=0)
  idata = sampler(seed=seed)
  if method.endswith("hmc"):
    assert max_rhat(idata) < 1.2
  else:
    assert max_rhat(idata) < 1.1


@pytest.mark.parametrize("method", METHODS)
def test_kwargs(method):
  model = bx.Model(log_density=lambda pt: -pt["x"]**2,
                   test_point={"x": jnp.array(1.)})
  sampler = getattr(model.mcmc, method)
  default_kwargs = sampler.get_kwargs()

  some_key, some_dict = list(default_kwargs.items())[0]
  assert isinstance(some_dict, dict)
  some_sub_key = list(some_dict.keys())[0]

  custom_kwargs = sampler.get_kwargs(**{some_sub_key: "did this work?"})
  assert custom_kwargs[some_key][some_sub_key] == "did this work?"


def test_tfp_bijectors():
  data = jax.random.normal(jax.random.PRNGKey((0)), shape=(100,))

  @tfd.JointDistributionCoroutineAutoBatched
  def tfd_model():
    x = yield tfd.Normal([0], 1., name="x")
    y = yield tfd.HalfNormal([1], name="y")
    yield tfd.Normal(x, y, name="z")

  pinned_model = tfd_model.experimental_pin(z=data)
  test_point = pinned_model.sample_unpinned(seed=jax.random.PRNGKey(1))

  model = bx.Model(
      log_density=pinned_model.log_prob,
      test_point=test_point,
      transform_fn=pinned_model.experimental_default_event_space_bijector())
  assert model.mcmc.blackjax_nuts.debug(seed=jax.random.PRNGKey(2),
                                        verbosity=10)
