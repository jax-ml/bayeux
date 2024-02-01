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

"""Initialization strategies."""
from typing import Callable

from bayeux._src.types import Point  # pylint: disable=g-importing-member
import jax
import jax.numpy as jnp


def uniform(
    *,
    test_point: Point,
    inverse_transform_fn: Callable[[Point], Point],
    transform_fn: Callable[[Point], Point],
    num_points: int,
    key: jax.Array
) -> Point:
  """A uniform initialization in unconstrained space between -2 and +2.

  Args:
    test_point: A PyTree that is used to determine the correct shape for
      initialization.
    inverse_transform_fn: A function that pulls the points back to R^n.
    transform_fn: A function that pushes points to the support of the log
      density.
    num_points: A number of points to initialize with.
    key: JAX PRNGkey.

  Returns:
    A batch of `num_points` with the same shape as `test_point`.
  """
  # We only need a shape here, but some bijectors may change the shape, so we
  # take the inverse transform, map over the shape, then transform back.
  untransformed = inverse_transform_fn(test_point)
  treedef = jax.tree_util.tree_structure(untransformed)
  keys = jax.random.split(key, treedef.num_leaves)
  draws = jax.tree_util.tree_map(
      lambda k, d: 4 * jax.random.uniform(  # pylint: disable=g-long-lambda
          key=k, shape=(num_points,) + jnp.shape(d)) - 2,
      jax.tree_util.tree_unflatten(treedef, keys), untransformed)

  return transform_fn(draws)
