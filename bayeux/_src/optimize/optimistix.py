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

"""optimistix specific code."""
from bayeux._src.optimize import shared
import optimistix


class _OptimistixOptimizer(shared.Optimizer):
  """Base class for optimistix optimizers."""

  def get_kwargs(self, **kwargs):
    kwargs = self.default_kwargs() | kwargs
    solver = getattr(optimistix, self.optimizer)
    minimise_kwargs = shared.get_optimizer_kwargs(
        optimistix.minimise, kwargs, ignore_required={"y0", "solver", "fn"})
    for k in minimise_kwargs:
      if k in kwargs:
        minimise_kwargs[k] = kwargs[k]
    extra_parameters = shared.get_extra_kwargs(kwargs)
    _ = extra_parameters.pop("num_iters")
    return {solver: shared.get_optimizer_kwargs(solver, kwargs),
            optimistix.minimise: minimise_kwargs,
            "extra_parameters": extra_parameters}

  def default_kwargs(self) -> dict[str, float]:
    return {"rtol": 1e-5, "atol": 1e-5}

  def _prep_args(self, seed, kwargs):
    fun, initial_state, apply_transform = super()._prep_args(seed, kwargs)
    def f(x, _):
      return fun(x)
    return f, initial_state, apply_transform

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    fun, initial_state, apply_transform = self._prep_args(seed, kwargs)

    solver_fn = getattr(optimistix, self.optimizer)
    def run(x0):
      solver = solver_fn(**kwargs[solver_fn])
      return optimistix.minimise(
          fn=fun,
          solver=solver,
          y0=x0,
          **kwargs[optimistix.minimise]).value
    chain_method = kwargs["extra_parameters"]["chain_method"]
    mapped_run = self._map_optimizer(chain_method, run)
    ret = mapped_run(initial_state)
    if apply_transform:
      return shared.OptimizerResults(
          params=self.transform_fn(ret), state=None, loss=None)
    else:
      return shared.OptimizerResults(ret, state=None, loss=None)


class BFGS(_OptimistixOptimizer):
  name = "optimistix_bfgs"
  optimizer = "BFGS"


class NelderMead(_OptimistixOptimizer):
  name = "optimistix_nelder_mead"
  optimizer = "NelderMead"


class NonlinearCG(_OptimistixOptimizer):
  name = "optimistix_nonlinear_cg"
  optimizer = "NonlinearCG"
