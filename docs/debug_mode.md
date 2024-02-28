# Debug Mode

Algorithms come with a built-in `debug` mode that attempts to fail quickly and in a manner that might help debug problems quickly. The signature for `debug` accepts `verbosity` and `catch_exceptions` arguments, as well as a `kwargs` dictionary that the user plans to pass to the algorithm itself.

## Default behavior

By default, debug mode will print a little description of what is happening, and whether the test passed. This can also be useful when unit testing your models, since the return value is whether all the tests passed!

```python
import bayeux as bx
import jax
import jax.numpy as jnp

normal_density = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.)

seed = jax.random.key(0)

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

## Do not catch exceptions

Often our models are bad because they don't even run. Debug mode aggresively catches exceptions, but you can disable that to make sure it is possible to use the model.

See if you can spot what is wrong with this model:

```python
bad_model = bx.Model(
    log_density=lambda x: jnp.sqrt(x['mean']),
    test_point=-1.)

bad_model.mcmc.numpyro_nuts.debug(seed=seed, catch_exceptions=False)

Checking test_point shape ✓
Computing test point log density ×
      ...
      1 bad_model = bx.Model(
----> 2     log_density=lambda x: jnp.sqrt(x['mean']),
      3     test_point=-1.)

TypeError: 'float' object is not subscriptable
```

## Changing verbosity

Debug mode also accepts a `verbosity` argument. The default is 2. We have a new subtly poorly specified `bad_model` with no outputs:

```python

bad_model = bx.Model(
    log_density=jnp.sqrt,
    test_point=-1.)

bad_model.mcmc.blackjax_nuts.debug(seed=seed, verbosity=0, kwargs={"num_chains": 17})

False
```

With `verbosity=1` there is a minimal output:

```python
bad_model.mcmc.blackjax_nuts.debug(seed=seed, verbosity=0, kwargs={"num_chains": 17})

✓ × ✓ ✓ ✓ × ✓ ✓ ×
False
```

With higher verbosity, we can see the actual outputs and perhaps diagnose the problem after seeing that the log density of the initial point is `nan`. We should have passed in a `transform=jnp.exp` or similar!:

```python
bad_model.mcmc.blackjax_nuts.debug(seed=seed, verbosity=3, kwargs={"num_chains": 17})

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
{<function window_adaptation at 0x14bd62b90>: {'algorithm': <class 'blackjax.mcmc.nuts.nuts'>,
                                               'initial_step_size': 1.0,
                                               'is_mass_matrix_diagonal': True,
                                               'logdensity_fn': <function constrain.<locals>.wrap_log_density.<locals>.wrapped at 0x15fb97880>,
                                               'progress_bar': False,
                                               'target_acceptance_rate': 0.8},
 'adapt.run': {'num_steps': 500},
 'extra_parameters': {'chain_method': 'vectorized',
                      'num_adapt_draws': 500,
                      'num_chains': 17,
                      'num_draws': 500,
                      'return_pytree': False},
 <class 'blackjax.mcmc.nuts.nuts'>: {'divergence_threshold': 1000,
                                     'integrator': <function generate_euclidean_integrator.<locals>.euclidean_integrator at 0x14bad0e50>,
                                     'logdensity_fn': <function constrain.<locals>.wrap_log_density.<locals>.wrapped at 0x15fb97880>,
                                     'max_num_doublings': 10,
                                     'step_size': 0.5}}
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

Even bigger numbers will give even more details.

## Fun mode

I mean, you're reading about debugging statistical models.

```python
bx.debug.FunMode.engaged = True

bad_model.mcmc.blackjax_nuts.debug(seed=seed, verbosity=1, kwargs={"num_chains": 17})

🌈 👎 💪 🙌 🚀 💀 🌈 ✓ ❌
False
```
