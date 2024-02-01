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

"""JAXopt specific code."""
import functools

from bayeux._src.optimize import shared
import jax
import jaxopt


_OPTIMIZER_FNS = {
    "BFGS": jaxopt.BFGS,
    "GradientDescent": jaxopt.GradientDescent,
    "LBFGS": jaxopt.LBFGS,
    "NonlinearCG": jaxopt.NonlinearCG,
}


class _JAXoptOptimizer(shared.Optimizer):
  """Base class for JAXopt samplers."""

  def get_kwargs(self, **kwargs):
    optimizer = _OPTIMIZER_FNS[self.optimizer]
    return {
        optimizer: shared.get_optimizer_kwargs(
            optimizer, kwargs, ignore_required={"fun"}),
        "extra_parameters": shared.get_extra_kwargs(kwargs)}

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    fun, initial_state, apply_transform = self._prep_args(seed, kwargs)

    optimizer_fn = _OPTIMIZER_FNS[self.optimizer]
    jaxopt_cls = optimizer_fn(fun=fun, **kwargs[optimizer_fn])  # pylint: disable=unexpected-keyword-arg
    num_iters = kwargs["extra_parameters"]["num_iters"]
    optimizer = functools.partial(_jaxopt_fit,
                                  jaxopt_cls=jaxopt_cls,
                                  num_iters=num_iters)
    chain_method = kwargs["extra_parameters"]["chain_method"]
    mapped_optimizer = self._map_optimizer(chain_method, optimizer)

    (res, _), loss = mapped_optimizer(initial_state)
    if apply_transform:
      params = self.transform_fn(res[0])
    else:
      params = res[0]
    return shared.OptimizerResults(
        params=params,
        state=res[1],
        loss=loss)


class BFGS(_JAXoptOptimizer):
  name = "jaxopt_bfgs"
  optimizer = "BFGS"


class GradientDescent(_JAXoptOptimizer):
  name = "jaxopt_gradient_descent"
  optimizer = "GradientDescent"


class LBFGS(_JAXoptOptimizer):
  name = "jaxopt_lbfgs"
  optimizer = "LBFGS"


class NonlinearCG(_JAXoptOptimizer):
  name = "jaxopt_nonlinear_cg"
  optimizer = "NonlinearCG"


def _jaxopt_fit(params, jaxopt_cls, num_iters):
  """JAXopt while loop.

  This is needed to run a `while` loop while collecting loss from the loop.

  Args:
    params: Initial position
    jaxopt_cls: A solver class in JAXopt, with methods `init_state`, `update`,
      and the attribute `tol`.
    num_iters: Maximum number of iterations. Loop switches to no-ops after the
      convergence criteria is met.

  Returns:
    (final params, final state), losses.
  """
  state = jaxopt_cls.init_state(params)

  def step(opt_state):
    params, state = jaxopt_cls.update(*opt_state)
    cond = state.error > jaxopt_cls.tol
    return (params, state), cond

  def fun(opt_state, _):
    val, cond = opt_state
    ret = jax.lax.cond(cond, step, lambda x: (x, False), val)
    return ret, ret[0][1].error

  init_val = (params, state)
  return jax.lax.scan(fun, (init_val, True), None, length=num_iters)
