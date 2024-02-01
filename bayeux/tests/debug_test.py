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

"""Tests for debug."""
import functools

import bayeux as bx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions


class ListWriter:

  def __init__(self):
    self.stream = []

  def write(self, string):
    self.stream.append(string)


@pytest.fixture
def writer_and_printer():
  writer = ListWriter()
  return writer, functools.partial(print, file=writer)


@pytest.fixture(params=['basic', 'linear'])
def model(request):
  if request.param == 'basic':
    return bx.Model(log_density=lambda x: -(x - 8)**2, test_point=1.)
  elif request.param == 'linear':
    np.random.seed(0)

    ndims = 5
    ndata = 100
    data = np.random.randn(ndata, ndims)
    w_ = np.random.randn(ndims)  # hidden
    noise_ = 0.1 * np.random.randn(ndata)  # hidden

    y_obs = data.dot(w_) + noise_

    @tfd.JointDistributionCoroutineAutoBatched
    def tfd_model():
      sigma = yield tfd.HalfNormal(1, name='sigma')
      w = yield tfd.Sample(tfd.Normal(0, sigma), sample_shape=ndims, name='w')
      yield tfd.Normal(jnp.einsum('...jk,...k->...j', data, w), 0.1, name='y')

    tfd_model = tfd_model.experimental_pin(y=y_obs)
    test_point = tfd_model.sample_unpinned(seed=jax.random.PRNGKey(1))
    transform = lambda pt: pt._replace(sigma=jnp.exp(pt.sigma))

    return bx.Model(
        tfd_model.unnormalized_log_prob,
        test_point,
        transform_fn=transform)


def test_output_no_verbosity(model, writer_and_printer):  # pylint: disable=redefined-outer-name
  writer, printer = writer_and_printer
  assert model.mcmc.blackjax_nuts.debug(
      jax.random.PRNGKey(0), verbosity=0, printer=printer)
  assert not writer.stream


def test_throws_exceptions():
  bad_model = bx.Model(lambda x: x['a'], test_point=1.)
  with pytest.raises(TypeError):
    bad_model.debug(seed=jax.random.PRNGKey(0), catch_exceptions=False)


def test_catches_exceptions():
  bad_model = bx.Model(lambda x: x['a'], test_point=1.)
  assert not bad_model.debug(seed=jax.random.PRNGKey(0),
                             verbosity=0,
                             catch_exceptions=True)


def test_output_low_verbosity(model, writer_and_printer):  # pylint: disable=redefined-outer-name
  writer, printer = writer_and_printer
  assert model.mcmc.blackjax_nuts.debug(
      jax.random.PRNGKey(0), verbosity=1, printer=printer)
  for j in writer.stream:
    assert (j == '✓') or (not j.strip())


@pytest.mark.parametrize('verbosity', [1, 2, 3, 4])
def test_output(model, writer_and_printer, verbosity):  # pylint: disable=redefined-outer-name
  _, printer = writer_and_printer
  assert model.mcmc.blackjax_nuts.debug(
      jax.random.PRNGKey(0), verbosity=verbosity, printer=printer)


def test_transform_change_shape():
  @tfd.JointDistributionCoroutineAutoBatched
  def tfp_model():
    probs = yield tfd.Dirichlet(np.ones(3))
    yield tfd.Multinomial(10, probs=probs, name='observed')

  conditioned = tfp_model.experimental_pin(observed=jnp.array([5, 2, 3]))
  draw = conditioned.sample_unpinned(seed=jax.random.PRNGKey(0))
  bij = conditioned.experimental_default_event_space_bijector()
  bx_model = bx.Model(
      conditioned.log_prob,
      draw,
      transform_fn=bij.forward,
      inverse_transform_fn=bij.inverse,
      inverse_log_det_jacobian=bij.inverse_log_det_jacobian)
  assert bx_model.optimize.optax_adam.debug(
      seed=jax.random.PRNGKey(0), verbosity=0)
