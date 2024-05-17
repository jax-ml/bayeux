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

"""optax specific code."""
import functools

from bayeux._src.optimize import shared
import jax
import optax
import optax.contrib


class _OptaxOptimizer(shared.Optimizer):
  """Base class for optax optimizers."""
  _base = optax

  def get_kwargs(self, **kwargs):
    kwargs = self.default_kwargs() | kwargs
    optimizer = getattr(self._base, self.optimizer)
    return {optimizer: shared.get_optimizer_kwargs(optimizer, kwargs),
            "extra_parameters": shared.get_extra_kwargs(kwargs)}

  def __call__(self, seed, **kwargs):
    kwargs = self.get_kwargs(**kwargs)
    fun, initial_state, apply_transform = self._prep_args(seed, kwargs)

    optimizer_fn = getattr(self._base, self.optimizer)
    optimizer = optimizer_fn(**kwargs[optimizer_fn])
    num_iters = kwargs["extra_parameters"]["num_iters"]
    optimizer = functools.partial(
        _optax_fit,
        neg_log_density=fun,
        optimizer=optimizer,
        num_iters=num_iters)
    chain_method = kwargs["extra_parameters"]["chain_method"]
    mapped_optimizer = self._map_optimizer(chain_method, optimizer)
    res, loss = mapped_optimizer(initial_state)
    if apply_transform:
      return shared.OptimizerResults(
          params=self.transform_fn(res[0]), state=res[1][0], loss=loss)
    else:
      return shared.OptimizerResults(*res, loss=loss)


class AdaBelief(_OptaxOptimizer):
  name = "optax_adabelief"
  optimizer = "adabelief"


class Adafactor(_OptaxOptimizer):
  name = "optax_adafactor"
  optimizer = "adafactor"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["decay_rate"] = 0.90
    kwargs["learning_rate"] = 1e-4
    return kwargs


class Adagrad(_OptaxOptimizer):
  name = "optax_adagrad"
  optimizer = "adagrad"


class Adam(_OptaxOptimizer):
  name = "optax_adam"
  optimizer = "adam"


class Adamw(_OptaxOptimizer):
  name = "optax_adamw"
  optimizer = "adamw"


class Adamax(_OptaxOptimizer):
  name = "optax_adamax"
  optimizer = "adamax"


class Adamaxw(_OptaxOptimizer):
  name = "optax_adamaxw"
  optimizer = "adamaxw"


class Amsgrad(_OptaxOptimizer):
  name = "optax_amsgrad"
  optimizer = "amsgrad"


class Fromage(_OptaxOptimizer):
  name = "optax_fromage"
  optimizer = "fromage"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class Lamb(_OptaxOptimizer):
  name = "optax_lamb"
  optimizer = "lamb"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs

# Too finicky
# class Lars(_OptaxOptimizer):
#   name = "optax_lars"
#   optimizer = "lars"


class Lion(_OptaxOptimizer):
  name = "optax_lion"
  optimizer = "lion"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class NoisySgd(_OptaxOptimizer):
  name = "optax_noisy_sgd"
  optimizer = "noisy_sgd"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class Novograd(_OptaxOptimizer):
  name = "optax_novograd"
  optimizer = "novograd"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs

# Cannot get this to pass tests.
# class OptimisticGradientDescent(_OptaxOptimizer):
#   name = "optax_optimistic_gradient_descent"
#   optimizer = "optimistic_gradient_descent"


# Re-enable this when we figure out how to pass a seed to an optimizer.
# class Dpsgd(_OptaxOptimizer):
#   name = "optax_dpsgd"
#   optimizer = "dpsgd"


class Radam(_OptaxOptimizer):
  name = "optax_radam"
  optimizer = "radam"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class Rmsprop(_OptaxOptimizer):
  name = "optax_rmsprop"
  optimizer = "rmsprop"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class ScheduleFree(_OptaxOptimizer):
  _base = optax.contrib
  name = "optax_schedule_free"
  optimizer = "schedule_free"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    base_optimizer = optax.adam(
        **shared.get_optimizer_kwargs(optax.adam, kwargs))
    kwargs["base_optimizer"] = base_optimizer
    return kwargs


class Sgd(_OptaxOptimizer):
  name = "optax_sgd"
  optimizer = "sgd"

  def default_kwargs(self) -> dict[str, float]:
    kwargs = super().default_kwargs()
    kwargs["learning_rate"] = 1e-4
    return kwargs


class Sm3(_OptaxOptimizer):
  name = "optax_sm3"
  optimizer = "sm3"


class Yogi(_OptaxOptimizer):
  name = "optax_yogi"
  optimizer = "yogi"


def _optax_fit(params, neg_log_density, optimizer, num_iters):
  opt_state = optimizer.init(params)

  def step(params_and_state, _):
    params, opt_state = params_and_state
    loss, grads = jax.value_and_grad(neg_log_density)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), loss

  init = (params, opt_state)
  return jax.lax.scan(step, init, xs=None, length=num_iters)
