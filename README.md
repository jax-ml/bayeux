# Bayeux

*Stitching together models and samplers*

[![Unittests](https://github.com/jax-ml/bayeux/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/jax-ml/bayeux/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/bayeux.svg)](https://badge.fury.io/py/bayeux)

The goal of `bayeux` is to allow users to write a model in JAX and use
best-in-class Bayesian inference methods on it. The API aims to be simple, self
descriptive, and helpful. The user is required to supply a (possibly
unnormalized) log density, along with a single
[pytree](https://jax.readthedocs.io/en/latest/pytrees.html), such that the log
density of that point is a finite scalar.

```python
import bayeux as bx

normal_model = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.)
```

Already, we can optimize this density, by supplying just a `jax.PRNGKey`:
```python
params, state = normal_model.optimize.jaxopt_lbfgs(seed=jax.random.PRNGKey(0))
```

In a similar way, we can run MCMC:

```python
idata = normal_model.mcmc.numpyro_nuts(seed=jax.random.PRNGKey(0))
```

A few things to note:

-   The `model.mcmc` namespace tab completes with the available MCMC algorithms,
    each of which will sample with sensible defaults after supplying a seed.
    Running `print(model.mcmc)` or `print(model.optimize)` will list available
    methods.
-   The return value for MCMC is an `arviz.InferenceData` object, which hooks
    into the [ArviZ library](https://python.arviz.org/) for analysis of the
    sampling.  More on InferenceData
    [here](https://python.arviz.org/en/stable/getting_started/XarrayforArviZ.html#xarray-for-arviz).
-   The return value for optimization is a `namedtuple` with fields `params` and
    `state`. The `params` will be the optimization results over a batch of
    particles (given by the `num_particles` argument). Given the variety of
    diagnostics for the provided optimization algorithms, the user may need to
    consult documentation for the given library to interpret the `state`.

In case we need to constrain some of the arguments, the `bx.Model` class accepts
an optional `transform_fn` argument. This should be an invertible JAX function
of a pytree real number into the support of the `log_density`. For example,

```python
half_normal_model = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.,
  transform_fn=jnp.exp)
```

## Using with TFP on JAX

```python
import numpy as np

## Generate linear data
np.random.seed(0)

ndims = 5
ndata = 100
X = np.random.randn(ndata, ndims)
w_ = np.random.randn(ndims)  # hidden
noise_ = 0.1 * np.random.randn(ndata)  # hidden

y_obs = X.dot(w_) + noise_

## Write a joint distribution in TFP and condition on the data
@tfd.JointDistributionCoroutineAutoBatched
def tfd_model():
  sigma = yield tfd.HalfNormal(1, name='sigma')
  w = yield tfd.Sample(tfd.Normal(0, sigma), sample_shape=ndims, name='w')
  yield tfd.Normal(jnp.einsum('...jk,...k->...j', X, w), 0.1, name='y')

tfd_model = tfd_model.experimental_pin(y=y_obs)
test_point = tfd_model.sample_unpinned(seed=jax.random.PRNGKey(1))
transform = lambda pt: pt._replace(sigma=jnp.exp(pt.sigma))

## Sample the model with bayeux
model = bx.Model(tfd_model.unnormalized_log_prob, test_point, transform_fn=transform)

idata = model.mcmc.numpyro_nuts(seed=jax.random.PRNGKey(2))

## Analyze with arviz
az.summary(idata)
```

|       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| sigma |  0.673 | 0.245 |    0.323 |     1.136 |       0.003 |     0.002 |       9016 |       5410 |       1 |
| w[0]  |  0.372 | 0.01  |    0.353 |     0.391 |       0     |     0     |       9402 |       6572 |       1 |
| w[1]  | -0.035 | 0.011 |   -0.055 |    -0.015 |       0     |     0     |      10148 |       6196 |       1 |
| w[2]  |  1.094 | 0.01  |    1.075 |     1.114 |       0     |     0     |      11007 |       6085 |       1 |
| w[3]  | -0.234 | 0.01  |   -0.254 |    -0.217 |       0     |     0     |       9604 |       6456 |       1 |
| w[4]  | -0.339 | 0.01  |   -0.359 |    -0.32  |       0     |     0     |      10838 |       6539 |       1 |

## Helpful features

### Debug mode

Each sampler has a `debug` method, which checks for common problems:

```python
normal_model.mcmc.numpyro_nuts.debug(seed=jax.random.PRNGKey(0))

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

You can additionally pass higher verbosity for more information, or keywords
that you plan to pass to the sampler. Here is a badly specified model:

```python
bad_model = bx.Model(
    log_density=jnp.sqrt,
    test_point=-1.)

model.mcmc.blackjax_nuts.debug(jax.random.PRNGKey(0),
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
{<function window_adaptation at 0x7fa4da751d80>: {'algorithm': <class 'blackjax.mcmc.nuts.nuts'>,
                                                  'initial_step_size': 1.0,
                                                  'is_mass_matrix_diagonal': True,
                                                  'progress_bar': False,
                                                  'target_acceptance_rate': 0.8},
 'extra_parameters': {'chain_method': 'vectorized',
                      'num_adapt_draws': 500,
                      'num_chains': 17,
                      'num_draws': 500},
 <class 'blackjax.mcmc.nuts.nuts'>: {'divergence_threshold': 1000,
                                     'integrator': <function velocity_verlet at 0x7fa4f0bbfac0>,
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

Comparing transformed log density to untransformed ×
Log density mismatch of up to nan
××××××××××

Computing gradients of transformed log density ×
The gradient contains NaNs! Initial gradients has shape
(17,)
××××××××××

False
```

Note that a verbosity of 0 will just return a boolean of whether the model seems
ok to run. The goal is to detect all possible problems before starting an
inference run -- please report errors that are not caught!

### Keyword inspection

Since `bayeux` aims to connect model specification with inference algorithms,
there may be many functions from different libraries that are called. A user can
inspect the functions and keywords via the `get_kwargs` argument:

```python
normal_model.mcmc.blackjax_nuts_pathfinder.get_kwargs()

{<function blackjax.adaptation.pathfinder_adaptation.pathfinder_adaptation>: {'algorithm': blackjax.mcmc.nuts.nuts,
  'initial_step_size': 1.0,
  'target_acceptance_rate': 0.8},
 blackjax.mcmc.nuts.nuts: {'divergence_threshold': 1000,
  'integrator': <function blackjax.mcmc.integrators.velocity_verlet>,
  'max_num_doublings': 10,
  'step_size': 0.01},
 'extra_parameters': {'chain_method': 'vectorized',
  'num_adapt_draws': 500,
  'num_chains': 8,
  'num_draws': 500}}
```

Note that some of the keys describe functions, but the calling conventions of
libraries are diverse enough that this is not quite dynamically generated
(though the defaults are automatically pulled from the libraries via
`inspect.getsignature`). Keywords can be overridden here, and all keywords
passed to the sampler will be just passed to `.get_kwargs`, so you can check
beforehand what arguments are being used, or save them for repeatability.

```python
normal_model.mcmc.numpyro_hmc.get_kwargs(target_accept_prob=0.99)

{numpyro.infer.hmc.HMC: {'adapt_mass_matrix': True,
  'adapt_step_size': True,
  'dense_mass': False,
  'find_heuristic_step_size': False,
  'forward_mode_differentiation': False,
  'init_strategy': <function numpyro.infer.initialization.init_to_uniform>,
  'inverse_mass_matrix': None,
  'kinetic_fn': None,
  'model': None,
  'num_steps': None,
  'regularize_mass_matrix': True,
  'step_size': 1.0,
  'target_accept_prob': 0.99,
  'trajectory_length': 6.283185307179586},
 numpyro.infer.mcmc.MCMC: {'chain_method': 'vectorized',
  'jit_model_args': False,
  'num_chains': 8,
  'num_samples': 1000,
  'num_warmup': 500,
  'postprocess_fn': None,
  'progress_bar': True,
  'thinning': 1}}
```

Note that *every* subkey matching a name gets replaced: for example, in
`blackjax`, multiple functions accept the same keyword arguments, and there are
reasonable reasons to want them to be different, but that's not possible here
yet. Also, some subkeys may not be honored. For example, `step_size` may get
adapted, and will overwrite the user-provided value.

*This is not an officially supported Google product.*
