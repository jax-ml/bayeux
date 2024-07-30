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

"""Utilities for debugging models."""
import dataclasses
import pprint
import random
from typing import Any, Callable, Mapping, Sequence, Type

from bayeux._src.types import Point  # pylint: disable=g-importing-member
import jax
import jax.numpy as jnp


class _Marks:
  """Unicode points for debug messages."""

  def __init__(self):
    self._goods = [
        "\N{CHECK MARK}", "\N{THUMBS UP SIGN}", "\N{DOG}", "\N{RAINBOW}",
        "\N{ROCKET}", "\N{FLEXED BICEPS}", "\N{PARTY POPPER}",
        "\N{PERSON RAISING BOTH HANDS IN CELEBRATION}"
    ]
    self._bads = ["\N{MULTIPLICATION SIGN}", "\N{CROSS MARK}",
                  "\N{THUMBS DOWN SIGN}", "\N{SKULL}"]
    self.engaged = False

  def good(self):
    if self.engaged:
      return random.choice(self._goods)
    else:
      return self._goods[0]

  def bad(self):
    if self.engaged:
      return random.choice(self._bads)
    else:
      return self._bads[0]


FunMode = _Marks()
_GOOD = FunMode.good
_BAD = FunMode.bad


DebugState = Mapping[str, Point]
Kwargs = Mapping[str, Any]
_Base = Any


@dataclasses.dataclass(frozen=True)
class DebugAttr:
  parents: Sequence["DebugAttr"]
  getter: Callable[[_Base, DebugState, Kwargs], DebugState]

  def __call__(self, model, debug_state, kwargs):
    if self not in debug_state:
      _ = [parent(model, debug_state, kwargs) for parent in self.parents]
      debug_state[self] = self.getter(model, debug_state, kwargs)
    return debug_state[self]


Seed = DebugAttr(parents=(), getter=lambda model, debug_state, _: None)

Verbosity = DebugAttr(parents=(), getter=lambda model, debug_state, _: None)

TestPoint = DebugAttr(
    parents=(), getter=lambda model, debug_state, _: model.test_point)

# pylint: disable=g-long-lambda
TestPointLogDensity = DebugAttr(
    parents=(TestPoint,),
    getter=lambda model, debug_state, _: model.log_density(
        debug_state[TestPoint]))

DefaultKwargs = DebugAttr(
    parents=(),
    getter=lambda model, debug_state, kwargs: model.get_kwargs(**kwargs))

InitialState = DebugAttr(
    parents=(Seed, DefaultKwargs),
    getter=lambda model, debug_state, _: model.get_initial_state(
        debug_state[Seed],
        num_chains=_get_num_chains(debug_state[DefaultKwargs])))

InitialStateLogDensity = DebugAttr(
    parents=(InitialState,),
    getter=lambda model, debug_state, _: jax.vmap(model.log_density)
    (debug_state[InitialState]))

UnconstrainedState = DebugAttr(
    parents=(InitialState,),
    getter=lambda model, debug_state, _: model.inverse_transform_fn(debug_state[
        InitialState]))

UnconstrainedLogDensityNoILDJ = DebugAttr(
    parents=(UnconstrainedState,),
    getter=lambda model, debug_state, _: jax.vmap(
        lambda x: model.log_density(model.transform_fn(x))
    )(debug_state[UnconstrainedState]),
)

UnconstrainedLogDensity = DebugAttr(
    parents=(UnconstrainedState,),
    getter=lambda model, debug_state, _: jax.vmap(
        model.constrained_log_density())(debug_state[UnconstrainedState]),
)

InitialGradientsNoILDJ = DebugAttr(
    parents=(UnconstrainedState,),
    getter=lambda model, debug_state, _: jax.vmap(
        jax.grad(model.transformed_negative_log_prob())
    )(debug_state[UnconstrainedState]),
)

InitialGradients = DebugAttr(
    parents=(UnconstrainedState,),
    getter=lambda model, debug_state, _: jax.vmap(
        jax.grad(model.constrained_log_density()))
    (debug_state[UnconstrainedState]),
)
# pylint: enable=g-long-lambda


def make_debug_print(printer):
  """Returns a debug print function."""
  def debug_print(*, msg, verbosity, success=None, end="\n", sep=" ",):
    if verbosity <= 1:
      if success is None:
        return
      msg = ""
      sep = ""
    if verbosity < 2:
      end = " "

    if success is None:
      msg = [msg]
    else:
      if success:
        msg = [_GOOD(), msg]
      else:
        msg = [_BAD(), msg]
    printer(*msg, end=end, sep=sep)
  return debug_print


@dataclasses.dataclass(frozen=True)
class DebugStep:
  """A single step of debugging."""
  description: str
  attrs: Sequence[DebugAttr]
  check_fn: Callable[[_Base, DebugState, Kwargs], bool]
  success_msg: Callable[[DebugState], str]
  failure_msgs: Sequence[Callable[[DebugState], str]] = tuple()
  exceptions: tuple[Type[BaseException], ...] = tuple()

  def _run_with_exceptions(
      self,
      *,
      fn,
      model,
      debug_state,
      verbosity,
      debug_print,
      kwargs,
      catch_exceptions: bool,
  ):
    """Runs a debug step and tries to catch any errors."""
    try:
      ret = fn(model, debug_state, kwargs)
      return ret, True, debug_state
    except self.exceptions as exc:  # pylint: disable=catching-non-exception
      msg = self.failure_msgs[self.exceptions.index(type(exc))]
      debug_print(
          msg=msg,
          verbosity=verbosity - 1,
          success=False)
      return None, False, debug_state
    except Exception as exc:
      debug_print(
          msg=f"Unexpected error!\n{exc!r}",
          verbosity=verbosity - 1,
          success=False)
      if catch_exceptions:
        debug_print(
            msg="Run with `catch_exceptions=False` to raise the full exception",
            verbosity=verbosity - 1)
        return None, False, debug_state
      raise exc

  def __call__(
      self,
      *,
      model,
      debug_state,
      verbosity,
      debug_print,
      kwargs,
      catch_exceptions: bool,
  ):
    debug_print(
        msg=self.description,
        verbosity=verbosity,
        end=" ")
    if any(attr not in debug_state for attr in self.attrs):
      for attr in self.attrs:
        _, ran, debug_state = self._run_with_exceptions(
            fn=attr,
            model=model,
            debug_state=debug_state,
            verbosity=verbosity,
            debug_print=debug_print,
            kwargs=kwargs,
            catch_exceptions=catch_exceptions)
        if not ran:
          return False, debug_state
    ret, ran, debug_state = self._run_with_exceptions(
        fn=self.check_fn,
        model=model,
        debug_state=debug_state,
        verbosity=verbosity,
        debug_print=debug_print,
        kwargs=kwargs,
        catch_exceptions=catch_exceptions)
    success = bool(ran and ret)
    if success:
      msg = self.success_msg(debug_state)
    else:
      msg = self.failure_msgs[-1](debug_state)
    debug_print(
        msg="",
        verbosity=verbosity - 1,
        success=success)
    debug_print(
        msg=msg,
        verbosity=verbosity - 1,
        end="\n")
    return success, debug_state


def _always_succeed(model, debug_state, kwargs):
  del model, debug_state, kwargs
  return True


def _get_shape(pytree):
  return jax.tree_util.tree_map(jnp.shape, pytree)


def pytree_to_msg(pytree, pytree_name, verbosity) -> str:
  if verbosity < 3:
    return ""
  elif verbosity == 3:
    return f"{pytree_name} has shape\n{_get_shape(pytree)!r}"
  elif verbosity > 3:
    return f"{pytree_name} is\n{pytree!r}"
  else:
    return ""

# pylint: disable=g-long-lambda
check_shapes = DebugStep(
    description="Checking test_point shape",
    attrs=(TestPoint,),
    check_fn=_always_succeed,
    success_msg=lambda state: pytree_to_msg(state[TestPoint], "Test point",
                                            state[Verbosity]))


def _check_test_point_log_density(model, debug_state, kwargs):
  del model, kwargs

  scalar_shape = ~bool(jnp.shape(debug_state[TestPointLogDensity]))
  is_real = ~jnp.isnan(debug_state[TestPointLogDensity])
  return scalar_shape and is_real


check_test_point_log_density = DebugStep(
    description="Computing test point log density",
    attrs=(TestPointLogDensity,),
    check_fn=_check_test_point_log_density,
    success_msg=lambda state:
    f"Test point has log density\n{state[TestPointLogDensity]!r}",
    failure_msgs=[
        lambda state:
        f"Test point has log density\n{state[TestPointLogDensity]!r}"
    ])

check_kwargs = DebugStep(
    description="Loading keyword arguments...",
    attrs=(DefaultKwargs,),
    check_fn=_always_succeed,
    success_msg=lambda state:
    f"Keyword arguments are\n{pprint.pformat(state[DefaultKwargs])}")

check_init = DebugStep(
    description="Checking it is possible to compute an initial state",
    attrs=(InitialState,),
    check_fn=_always_succeed,
    success_msg=lambda state: pytree_to_msg(state[InitialState],
                                            "Initial state", state[Verbosity]))


check_init_nan = DebugStep(
    description="Checking initial state is has no NaN",
    attrs=(InitialState,),
    check_fn=lambda model, debug_state, kwargs: ~_has_nans(
        debug_state[InitialState]),
    success_msg=lambda state: "No nans detected!",
    failure_msgs=[
        lambda state: pytree_to_msg(
            state[InitialState], "Initial state has nans!",
            state[Verbosity])])


def _check_init_log_density(model, debug_state, kwargs):
  del model, kwargs
  expected_shape = (_get_num_chains(debug_state[DefaultKwargs]),)
  right_shape = jnp.shape(debug_state[InitialStateLogDensity]) == expected_shape
  has_nans = ~_has_nans(debug_state[InitialStateLogDensity])
  return right_shape and has_nans


check_init_log_density = DebugStep(
    description="Computing initial state log density",
    attrs=(InitialStateLogDensity, DefaultKwargs),
    check_fn=_check_init_log_density,
    success_msg=lambda state: pytree_to_msg(state[
        InitialStateLogDensity], "Initial state log density", state[Verbosity]),
    failure_msgs=[
        lambda state:
        f"Initial state has log density\n{state[InitialStateLogDensity]!r}"
    ])


def _get_unconstrained_state(model, debug_state, kwargs):
  del kwargs
  debug_state["unconstrained_state"] = model.inverse_transform_fn(
      debug_state["initial_state"])
  return debug_state


def _check_unconstrained_state(model, debug_state, kwargs):
  del model, kwargs
  if _has_nans(debug_state[UnconstrainedState]):
    debug_state["_msg"] = (
        "Some points are NaN. Check `model.inverse_transform_fn`, which is "
        "automatically generated by `oryx.core.inverse(model.transform_fn)`, "
        "and should map from the support of the log density to R^n. Maybe "
        "the opposite direction transform was provided?")
    return False
  return True


check_transform = DebugStep(
    description="Transforming model to R^n",
    attrs=(UnconstrainedState,),
    check_fn=_check_unconstrained_state,
    exceptions=(NotImplementedError,),
    success_msg=lambda state: pytree_to_msg(state[
        UnconstrainedState], "Transformed state", state[Verbosity]),
    failure_msgs=[
        lambda state:
        "Automatic inverse and ildj not supported for this model.",
        lambda state: state["_msg"]
    ])


def _check_transformed_log_density_no_ildj(model, debug_state, kwargs):
  del model, kwargs
  expected_shape = (_get_num_chains(debug_state[DefaultKwargs]),)
  return jnp.shape(debug_state[UnconstrainedLogDensityNoILDJ]) == expected_shape


def _check_transformed_log_density(model, debug_state, kwargs):
  del model, kwargs
  expected_shape = (_get_num_chains(debug_state[DefaultKwargs]),)
  return jnp.shape(debug_state[UnconstrainedLogDensity]) == expected_shape


check_transformed_log_density_no_ildj = DebugStep(
    description="Computing transformed state log density shape",
    attrs=(UnconstrainedLogDensityNoILDJ, DefaultKwargs),
    check_fn=_check_transformed_log_density_no_ildj,
    success_msg=lambda state: pytree_to_msg(state[
        UnconstrainedLogDensityNoILDJ], "Transformed state log density", state[
            Verbosity]),
    failure_msgs=[
        lambda state: pytree_to_msg(state[UnconstrainedLogDensityNoILDJ],
                                    "Transformed state log density", state[
                                        Verbosity])
    ])

check_transformed_log_density = DebugStep(
    description="Computing transformed state log density shape",
    attrs=(UnconstrainedLogDensity, DefaultKwargs),
    check_fn=_check_transformed_log_density,
    success_msg=lambda state: pytree_to_msg(state[
        UnconstrainedLogDensity], "Transformed state log density", state[
            Verbosity]),
    failure_msgs=[
        lambda state: pytree_to_msg(state[UnconstrainedLogDensity],
                                    "Transformed state log density", state[
                                        Verbosity])
    ])


compute_gradients_no_ildj = DebugStep(
    description="Computing gradients of transformed log density",
    attrs=(InitialGradientsNoILDJ,),
    check_fn=lambda model, debug_state, _: ~_has_nans(
        debug_state[InitialGradientsNoILDJ]),
    success_msg=lambda state: pytree_to_msg(
        state[InitialGradientsNoILDJ], "Initial gradient", state[Verbosity]
    ),
    failure_msgs=[
        lambda state: "The gradient contains NaNs! "
        + pytree_to_msg(
            state[InitialGradientsNoILDJ], "Initial gradients", state[Verbosity]
        )
    ],
)
compute_gradients = DebugStep(
    description="Computing gradients of transformed log density",
    attrs=(InitialGradients,),
    check_fn=lambda model, debug_state, _: ~_has_nans(
        debug_state[InitialGradients]),
    success_msg=lambda state: pytree_to_msg(
        state[InitialGradients], "Initial gradient", state[Verbosity]
    ),
    failure_msgs=[
        lambda state: "The gradient contains NaNs! "
        + pytree_to_msg(
            state[InitialGradients], "Initial gradients", state[Verbosity]
        )
    ],
)
# pylint: enable=g-long-lambda


class ModelDebug:
  """Controller for running debug steps."""

  def __init__(self, model, seed, debug_steps, kwargs):
    self.model = model
    self.seed = seed
    self.debug_steps = debug_steps
    self.kwargs = kwargs

  def __call__(
      self, verbosity, catch_exceptions: bool, printer: Callable[..., Any]
  ):
    if verbosity <= 0:
      printer = lambda *args, **kwargs: None
    debug_state = {Seed: self.seed, Verbosity: verbosity}
    debug_print = make_debug_print(printer)
    all_passed = True
    for debug_step in self.debug_steps:
      passed, debug_state = debug_step(
          model=self.model,
          debug_state=debug_state,
          verbosity=verbosity,
          debug_print=debug_print,
          kwargs=self.kwargs,
          catch_exceptions=catch_exceptions)
      all_passed &= passed
      if verbosity < 3:
        msg = ""
      else:
        msg = "".join([_BAD, _GOOD][passed]() for _ in range(10)) + "\n"
      debug_print(msg=msg, verbosity=verbosity)
    return all_passed


def debug(model, seed, verbosity, printer, kwargs, catch_exceptions: bool):
  """Debugger that includes the inverse log det jacobian."""
  checkers = [
      check_shapes,
      check_test_point_log_density,
      check_kwargs,
      check_init,
      check_init_nan,
      check_init_log_density,
      check_transform,
      check_transformed_log_density,
      compute_gradients]
  if kwargs is None:
    kwargs = {}
  return ModelDebug(model, seed, checkers, kwargs)(
      verbosity=verbosity, catch_exceptions=catch_exceptions, printer=printer)


def debug_no_ildj(
    model, seed, verbosity, printer, kwargs, catch_exceptions: bool
):
  """Debugger for models with no inverse log det jacobian correction."""
  checkers = [
      check_shapes,
      check_test_point_log_density,
      check_kwargs,
      check_init,
      check_init_nan,
      check_init_log_density,
      check_transform,
      check_transformed_log_density_no_ildj,
      compute_gradients_no_ildj]
  if kwargs is None:
    kwargs = {}
  return ModelDebug(model, seed, checkers, kwargs)(
      verbosity=verbosity, catch_exceptions=catch_exceptions, printer=printer)


def _get_num_chains(default_kwargs):
  for v in default_kwargs.values():
    for key in ("num_chains", "num_particles", "batch_size", "chains"):
      if key in v:
        return v[key]
  raise KeyError("No `num_chains` in default kwargs!")


def _has_nans(pytree):
  return jax.tree_util.tree_reduce(lambda t, c: t and jnp.any(jnp.isnan(c)),
                                   pytree, True)
