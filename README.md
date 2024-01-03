# Bayeux

*Stitching together models and samplers*

[![Unittests](https://github.com/jax-ml/bayeux/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/jax-ml/bayeux/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/bayeux_ml.svg)](https://badge.fury.io/py/bayeux_ml)

`bayeux` lets you write a probabilistic model in JAX and immediately have access to state-of-the-art inference methods. The API aims to be **simple**, **self descriptive**, and **helpful**. Simply provide a log density function (which doesn't even have to be normalized), along with a single point (specified as a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html)) where that log density is finite. Then let `bayeux` do the rest!

## Installation

```bash
pip install bayeux-ml
```
## Quickstart

We define a model by providing a log density in JAX. This could be defined using a probabilistic programming language (PPL) like numpyro, PyMC, TFP, distrax, oryx, coix, or directly in JAX.

```python
import bayeux as bx
import jax

normal_density = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.)

seed = jax.random.PRNGKey(0)
```

## Simple
Every inference algorithm in `bayeux` will (try to) run with just a seed as an argument:

```python
opt_results = normal_density.optimize.optax_adam(seed=seed)
# OR!
idata = normal_density.mcmc.numpyro_nuts(seed=seed)
# OR!
surrogate_posterior, loss = normal_density.vi.tfp_factored_surrogate_posterior(seed=seed)
```

An (only rarely) optional third argument to `bx.Model` is `transform_fn`, which maps a real number to the support of the distribution. The [oryx](https://github.com/jax-ml/oryx) library is used to automatically compute the inverse and Jacobian determinants for changes of variables, but the user can supply these if known.

```python
half_normal_density = bx.Model(
    lambda x: -x*x,
    test_point=1.,
    transform_fn=jax.nn.softplus)
```

## Self descriptive

Since `bayeux` is built on top of other fantastic libraries, it tries not to get in the way of them. Each algorithm has a `.get_kwargs()` method that tells you how it will be called, and what functions are being called:

```python
normal_density.optimize.jaxopt_bfgs.get_kwargs()

{jaxopt._src.bfgs.BFGS: {'value_and_grad': False,
  'has_aux': False,
  'maxiter': 500,
  'tol': 0.001,
  'stepsize': 0.0,
  'linesearch': 'zoom',
  'linesearch_init': 'increase',
  'condition': None,
  'maxls': 30,
  'decrease_factor': None,
  'increase_factor': 1.5,
  'max_stepsize': 1.0,
  'min_stepsize': 1e-06,
  'implicit_diff': True,
  'implicit_diff_solve': None,
  'jit': True,
  'unroll': 'auto',
  'verbose': False},
 'extra_parameters': {'chain_method': 'vectorized',
  'num_particles': 8,
  'num_iters': 1000,
  'apply_transform': True}}
```

If you pass an argument into `.get_kwargs()`, this will also tell you what will be passed on to the actual algorithms.

```
normal_density.mcmc.blackjax_nuts.get_kwargs(
    num_chains=5,
    target_acceptance_rate=0.99)

{<blackjax.adaptation.window_adaptation.window_adaptation: {'is_mass_matrix_diagonal': True,
  'initial_step_size': 1.0,
  'target_acceptance_rate': 0.99,
  'progress_bar': False,
  'algorithm': blackjax.mcmc.nuts.nuts},
 blackjax.mcmc.nuts.nuts: {'max_num_doublings': 10,
  'divergence_threshold': 1000,
  'integrator': blackjax.mcmc.integrators.velocity_verlet,
  'step_size': 0.01},
 'extra_parameters': {'chain_method': 'vectorized',
  'num_chains': 5,
  'num_draws': 500,
  'num_adapt_draws': 500,
  'return_pytree': False}}
```

A full list of available algorithms and how to call them can be seen with

```python
print(normal_density)

mcmc
  .blackjax_hmc
  .blackjax_nuts
  .blackjax_hmc_pathfinder
  .blackjax_nuts_pathfinder
  .numpyro_hmc
  .numpyro_nuts
optimize
  .jaxopt_bfgs
  .jaxopt_gradient_descent
  .jaxopt_lbfgs
  .jaxopt_nonlinear_cg
  .optax_adabelief
  .optax_adafactor
  .optax_adagrad
  .optax_adam
  .optax_adamw
  .optax_adamax
  .optax_amsgrad
  .optax_fromage
  .optax_lamb
  .optax_lion
  .optax_noisy_sgd
  .optax_novograd
  .optax_radam
  .optax_rmsprop
  .optax_sgd
  .optax_sm3
  .optax_yogi
vi
  .tfp_factored_surrogate_posterior

```

## Helpful

Algorithms come with a built-in `debug` mode that attempts to fail quickly and in a manner that might help debug problems quickly. The signature for `debug` accepts `verbosity` and `catch_exceptions` arguments, as well as a `kwargs` dictionary that the user plans to pass to the algorithm itself.

```python
normal_density.mcmc.numpyro_nuts.debug(seed=seed)

Checking test_point shape ✓
Computing test point log density ✓
Loading keyword arguments... ✓
Checking it is possible to compute an initial state ✓
Checking initial state is has no NaN ✓
Computing initial state log density ✓
Transforming model to R^n ✓
Computing transformed state log density shape ✓
Comparing transformed log density to untransformed ✓
Computing gradients of transformed log density ✓
True
```

Here is an example of a bad model with a higher verbosity:
```python
import jax.numpy as jnp

bad_model = bx.Model(
    log_density=jnp.sqrt,
    test_point=-1.)

bad_model.mcmc.blackjax_nuts.debug(jax.random.PRNGKey(0),
                                   verbosity=3, kwargs={"num_chains": 17})

Checking test_point shape ✓
Test point has shape
()
✓✓✓✓✓✓✓✓✓✓

Computing test point log density ×
Test point has log density
Array(nan, dtype=float32, weak_type=True)
××××××××××

Loading keyword arguments... ✓
Keyword arguments are
{<function window_adaptation at 0x77feef9308b0>: {'algorithm': <class 'blackjax.mcmc.nuts.nuts'>,
                                                  'initial_step_size': 1.0,
                                                  'is_mass_matrix_diagonal': True,
                                                  'progress_bar': False,
                                                  'target_acceptance_rate': 0.8},
 'extra_parameters': {'chain_method': 'vectorized',
                      'num_adapt_draws': 500,
                      'num_chains': 17,
                      'num_draws': 500,
                      'return_pytree': False},
 <class 'blackjax.mcmc.nuts.nuts'>: {'divergence_threshold': 1000,
                                     'integrator': <function velocity_verlet at 0x77feefbf4b80>,
                                     'max_num_doublings': 10,
                                     'step_size': 0.01}}
✓✓✓✓✓✓✓✓✓✓

Checking it is possible to compute an initial state ✓
Initial state has shape
(17,)
✓✓✓✓✓✓✓✓✓✓

Checking initial state is has no NaN ✓
No nans detected!
✓✓✓✓✓✓✓✓✓✓

Computing initial state log density ×
Initial state has log density
Array([1.2212421 ,        nan,        nan, 1.4113309 ,        nan,
              nan,        nan,        nan,        nan,        nan,
       0.5912253 ,        nan,        nan,        nan, 0.65457666,
              nan,        nan], dtype=float32)
××××××××××

Transforming model to R^n ✓
Transformed state has shape
(17,)
✓✓✓✓✓✓✓✓✓✓

Computing transformed state log density shape ✓
Transformed state log density has shape
(17,)
✓✓✓✓✓✓✓✓✓✓

Computing gradients of transformed log density ×
The gradient contains NaNs! Initial gradients has shape
(17,)
××××××××××

False
```


*This is not an officially supported Google product.*