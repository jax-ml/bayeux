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

"""Tests for the optimization API."""
import bayeux as bx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
METHODS = [getattr(bx.optimize, k).name for k in bx.optimize.__all__]


@pytest.fixture(scope="session")
def linear_model():
  np.random.seed(0)

  ndims = 5
  ndata = 100
  data = np.random.randn(ndata, ndims)
  w_ = np.random.randn(ndims)  # hidden
  noise_ = 0.1 * np.random.randn(ndata)  # hidden

  y_obs = data.dot(w_) + noise_

  @tfd.JointDistributionCoroutineAutoBatched
  def tfd_model():
    w = yield tfd.Sample(tfd.Normal(0, 1.), sample_shape=ndims, name="w")
    yield tfd.Normal(jnp.einsum("...jk,...k->...j", data, w), 0.1, name="y")

  tfd_model = tfd_model.experimental_pin(y=y_obs)
  test_point = tfd_model.sample_unpinned(seed=jax.random.PRNGKey(1))
  solution = np.linalg.solve(data.T @ data + 0.01 * np.eye(5), data.T @ y_obs)

  return solution, bx.Model(tfd_model.unnormalized_log_prob,
                            test_point=test_point)


@pytest.mark.parametrize("method", METHODS)
def test_optimizers(method, linear_model):  # pylint: disable=redefined-outer-name
  solution, linear_model = linear_model
  optimizer = getattr(linear_model.optimize, method)
  seed = jax.random.PRNGKey(0)
  if method in {
      "optax_radam",
      "optax_adafactor",
      "optax_fromage",
      "optax_lamb",
      "optax_lion",
      "optax_novograd",
      "optax_rmsprop",
  }:
    num_iters = 100_000
  else:
    num_iters = 1_000

  if method.startswith("optimistix"):
    num_iters = 10_000  # should stop automatically before then
    atol = 0.2
  else:
    atol = 1e-2

  assert optimizer.debug(seed=seed, verbosity=0)
  num_particles = 6
  params = optimizer(
      seed=seed,
      num_particles=num_particles,
      num_iters=num_iters,
      atol=atol,
      max_steps=num_iters,
      throw=False).params
  expected = np.repeat(solution[..., np.newaxis], num_particles, axis=-1).T

  if method not in {
      "optax_adafactor",
      "optax_adagrad",
      "optax_sm3",
      "optimistix_bfgs",
      "optimistix_chord",
      "optimistix_nelder_mead"}:
    np.testing.assert_allclose(expected, params.w, atol=atol)


def test_initial_state():
  num_particles = 10
  model = bx.Model(
      jnp.sin, test_point=0.0, initial_state=100.0 * jnp.ones(num_particles))
  res = model.optimize.optax_adam(seed=jax.random.PRNGKey(0),
                                  num_particles=num_particles)
  # Just make sure optimization goes to something high.
  np.testing.assert_array_less(50 * np.ones(num_particles), res.params)
