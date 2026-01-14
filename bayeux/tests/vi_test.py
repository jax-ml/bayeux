# Copyright 2025 The bayeux Authors.
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

"""Tests for VI."""
import bayeux as bx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
METHODS = [getattr(bx.vi, k).name for k in bx.vi.__all__]


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
def test_vi_algorithms(method, linear_model):  # pylint: disable=redefined-outer-name
  solution, linear_model = linear_model
  vi_algorithm = getattr(linear_model.vi, method)
  seed = jax.random.PRNGKey(0)
  fit_seed, draw_seed = jax.random.split(seed)

  assert vi_algorithm.debug(seed=seed, verbosity=3, catch_exceptions=False)
  batch_size = 8
  surrogate_dist, loss = vi_algorithm(seed=fit_seed, batch_size=batch_size)
  np.testing.assert_array_less(loss[:, -1], loss[:, 0])

  draws = surrogate_dist.sample(seed=draw_seed, sample_shape=100)

  np.testing.assert_allclose(solution,
                             jnp.mean(draws.w, axis=(0, 1)),
                             atol=1e-2)
