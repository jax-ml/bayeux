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

"""TensorF[riendly] Probability specific code."""
import functools

from bayeux._src import shared
import jax
import jax.numpy as jnp
import optax
import oryx
import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors


class Custom(tfb.Bijector):
  """Custom tfp bijector."""

  def __init__(self, bx_model):
    super().__init__(
        forward_min_event_ndims=jax.tree.map(jnp.ndim, bx_model.test_point))
    self.bx_model = bx_model

  def _forward(self, x):
    return self.bx_model.transform_fn(x)

  def _inverse(self, y):
    return self.bx_model.inverse_transform_fn(y)

  def _inverse_log_det_jacobian(self, y):
    return self.bx_model.inverse_log_det_jacobian(y)

  def _forward_log_det_jacobian(self, x):
    return -self.inverse_log_det_jacobian(self.forward(x))

  def _forward_event_shape_tensor(self, input_shape):
    return jax.tree.map(jnp.shape,
                        self._forward(jax.tree.map(jnp.ones, input_shape)))

  def _inverse_event_shape_tensor(self, output_shape):
    return jax.tree.map(jnp.shape,
                        self._inverse(jax.tree.map(jnp.ones, output_shape)))


def get_fit_kwargs(log_density, kwargs):
  """Get keyword arguments for fitting VI."""
  fit_kwargs, fit_required = shared.get_default_signature(
      tfp.vi.fit_surrogate_posterior_stateless)
  fit_kwargs.pop("seed")
  fit_kwargs["optimizer"] = optax.adam(learning_rate=0.01)
  fit_kwargs["target_log_prob_fn"] = jax.vmap(log_density)

  fit_kwargs = {
      "sample_size": 16,
      "num_steps": 2_000,
      "jit_compile": True,
  } | fit_kwargs
  fit_kwargs.update({k: kwargs[k] for k in fit_required if k in kwargs})
  fit_required = (fit_required
                  - set(fit_kwargs.keys())
                  - {"build_surrogate_posterior_fn", "initial_parameters"})
  if fit_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(fit_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords.")
  return fit_kwargs


def get_build_kwargs(event_shape, bijector, kwargs):
  """Get keyword arguments for building VI."""
  build_kwargs, build_required = shared.get_default_signature(
      tfp.experimental.vi.build_factored_surrogate_posterior_stateless)
  build_kwargs["event_shape"] = event_shape
  build_kwargs["bijector"] = bijector
  build_kwargs.update({k: kwargs[k] for k in build_required if k in kwargs})
  build_required = build_required - set(build_kwargs.keys())
  if build_required:
    raise ValueError(
        "Unexpected required arguments: "
        f"{','.join(build_required)}. Probably file a bug, but "
        "you can try to manually supply them as keywords."
    )
  return build_kwargs


class Factored(shared.Base):
  """Base class for TFP variational inference."""
  name = "tfp_factored_surrogate_posterior"

  def get_kwargs(self, **kwargs):
    return {
        tfp.experimental.vi.build_factored_surrogate_posterior_stateless: (
            get_build_kwargs(
                jax.tree.map(jnp.shape, self.test_point),
                self.constraining_bijector(),
                kwargs)),
        tfp.vi.fit_surrogate_posterior_stateless: get_fit_kwargs(
            self.log_density, kwargs),
        "extra_parameters": {
            "chain_method": kwargs.get("chain_method", "vectorized"),
            "batch_size": kwargs.get("batch_size", 16),
        }}

  def constraining_bijector(self):
    if self.inverse_log_det_jacobian is None:
      self.inverse_log_det_jacobian = oryx.core.ildj(self.transform_fn)
    return Custom(self)

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    batch_size = kwargs["extra_parameters"]["batch_size"]
    # TODO(colcarroll): The initialization is being done by TFP here. It would
    # be nice to put a vmap-friendly initialization in.
    init_fn, build_surrogate_posterior_fn = (
        tfp.experimental.vi.build_factored_surrogate_posterior_stateless(
            **kwargs[
                tfp.experimental.vi.build_factored_surrogate_posterior_stateless
            ]))

    fit_fn = _make_tfp_fit(
        build_surrogate_posterior_fn=build_surrogate_posterior_fn,
        init_fn=init_fn,
        fit_kwargs=kwargs[tfp.vi.fit_surrogate_posterior_stateless])
    chain_method = kwargs["extra_parameters"]["chain_method"]
    if chain_method == "vectorized":
      mapped_fit = jax.vmap(fit_fn)
    elif chain_method == "parallel":
      mapped_fit = jax.pmap(fit_fn)
    elif chain_method == "sequential":
      mapped_fit = functools.partial(jax.tree.map, fit_fn)
    else:
      raise ValueError(f"Chain method {chain_method} not supported.")

    map_seed = jax.random.split(seed, batch_size)
    params, loss = mapped_fit(seed=map_seed)
    return build_surrogate_posterior_fn(params), loss


def _make_tfp_fit(
    build_surrogate_posterior_fn, init_fn, fit_kwargs
):
  def run_vi(seed):
    init_seed, opt_seed = jax.random.split(seed)
    return tfp.vi.fit_surrogate_posterior_stateless(
        build_surrogate_posterior_fn=build_surrogate_posterior_fn,
        initial_parameters=init_fn(init_seed),
        seed=opt_seed,
        **fit_kwargs)

  return run_vi
