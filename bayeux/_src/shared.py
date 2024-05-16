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

"""Shared functionality for MCMC sampling."""

import dataclasses
import functools
import inspect
from typing import Any, Callable, Optional

from bayeux._src import debug
from bayeux._src import initialization
from bayeux._src import types
import jax
import jax.numpy as jnp
import oryx


def map_fn(chain_method, fn):
  if chain_method == "parallel":
    return jax.pmap(fn)
  elif chain_method == "vectorized":
    return jax.vmap(fn)
  elif chain_method == "sequential":
    return functools.partial(jax.tree.map, fn)
  raise ValueError(f"Chain method {chain_method} not supported.")


def update_with_kwargs(
    defaults: dict[str, Any],
    *,
    reqd: Optional[set[str]] = None,
    kwargs: dict[str, Any],
):
  """Updates a defaults dictionary, overwriting keys, but not adding new ones.

  This updates `defaults` in-place.

  Args:
    defaults: Dictionary of arguments.
    reqd: Set of allowed new keys.
    kwargs: Dictionary of key/values to overwrite in default.

  Returns:
    The defaults dictionary. Keys will be the same as the existing ones, or from
    the set `reqd`. Note that this is also updated in-place.
  """
  if reqd is None:
    reqd = set()
  defaults.update(
      (k, kwargs[k]) for k in (defaults.keys() | reqd) & kwargs.keys())
  return defaults


def _default_init(
    *,
    initial_state,
    test_point,
    inverse_transform_fn,
    transform_fn,
    num_points,
    key
):
  """Initialization in case there is no explicit init provided."""
  if initial_state is None:
    return initialization.uniform(
        test_point=test_point,
        inverse_transform_fn=inverse_transform_fn,
        transform_fn=transform_fn,
        num_points=num_points,
        key=key)
  else:
    return initial_state


def constrain(
    transform_fn: types.JAXFn,
    inverse_log_det_jacobian: Optional[types.JAXFn] = None,
) -> Callable[[types.LogDensityFn], types.LogDensityFn]:
  """Returns a log density function that operates in an unconstrained space.

  Adapted from oryx (https://github.com/jax-ml/oryx)

  Args:
    transform_fn: Constraining bijector, mapping from R^n to the support of the
      target log density.
    inverse_log_det_jacobian: Optional inverse log det jacobian, if known.
  """
  if inverse_log_det_jacobian is None:
    inverse_log_det_jacobian = oryx.core.ildj(transform_fn)

  def wrap_log_density(target_log_density):
    def wrapped(args):
      mapped_args = transform_fn(args)
      ildjs = inverse_log_det_jacobian(mapped_args)
      return target_log_density(mapped_args) - jnp.sum(
          jnp.array(jax.tree_util.tree_leaves(ildjs)))

    return wrapped

  return wrap_log_density


def get_default_signature(fn):
  defaults = {}
  required = set()
  for key, val in inspect.signature(fn).parameters.items():
    if val.default is inspect.Signature.empty:
      required.add(key)
    else:
      defaults[key] = val.default
  return defaults, required


def _nothing(x: types.Point) -> types.Point:
  return x


@dataclasses.dataclass
class Base:
  """Base class for MCMC sampling."""
  log_density: types.LogDensityFn
  test_point: types.Point
  transform_fn: types.JAXFn = _nothing
  inverse_transform_fn: Optional[types.JAXFn] = None
  inverse_log_det_jacobian: Optional[types.JAXFn] = None

  initial_state: Optional[types.Point] = None

  def get_initial_state(self, key, num_chains=8):
    return _default_init(
        initial_state=self.initial_state,
        test_point=self.test_point,
        inverse_transform_fn=self.inverse_transform_fn,
        transform_fn=self.transform_fn,
        num_points=num_chains,
        key=key)

  def get_kwargs(self, **kwargs):
    raise NotImplementedError()

  def constrained_log_density(self):
    return constrain(
        self.transform_fn,
        inverse_log_det_jacobian=self.inverse_log_det_jacobian,
    )(self.log_density)

  def debug(
      self,
      seed,
      verbosity=2,
      catch_exceptions: bool = True,
      printer=print,
      kwargs=None,
  ):
    return debug.debug(
        self,
        seed,
        verbosity=verbosity,
        catch_exceptions=catch_exceptions,
        printer=printer,
        kwargs=kwargs,
    )
