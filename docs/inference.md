# Building models

The two main contracts `bayeux` has are that
1. You can specify a model using a log density, a test point, and a transformation (the transformation defaults to an identity, but that is rarely what you want)
2. Every inference algorithm in `bayeux` will (try to) run with just a seed as an argument.

## Specifying a model

In case you have a scalar model, there is no need to normalize the density.

```python
import bayeux as bx
import jax
import numpy as np

normal_density = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.)
```

Suppose we have a bunch of observations of a normal distribution, and we want to infer the mean and scale. Maybe we write this down by hand, putting a prior of N(0, 10) on the mean and half normal with scale 10 on the scale:

```python
points = 3 * np.random.randn(100) - 10

def log_density(pt):
    log_prior = -(pt['loc'] ** 2 + pt['scale']**2) / 200.
    log_likelihood = jnp.sum(jst.norm.logpdf(points, loc=pt['loc'], scale=pt['scale']))
    return log_prior + log_likelihood
```

We additionally need to restrict the scale to be positive. A [softplus](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus) is useful for this:

```python
def transform_fn(pt):
  return {'loc': pt['loc'], 'scale': jax.nn.softplus(pt['scale'])}
```

The [oryx](https://github.com/jax-ml/oryx) library is used to automatically compute the inverse and Jacobian determinants for changes of variables, but the user can supply these if known.

Then we can get the model:
```python
model = bx.Model(
    log_density=log_density,
    test_point={'loc': 0., 'scale': 1.},
    transform_fn=transform_fn)

opt = model.optimize.optax_adam(seed=seed, num_iters=10000)
opt.params

{'loc': Array([-9.428163, -9.428162, -9.428163, -9.428162, -9.428165, -9.428163,
        -9.428163, -9.428164], dtype=float32),
 'scale': Array([2.9746027, 2.9746041, 2.9746022, 2.9746022, 2.9745977, 2.9746022,
        2.9746027, 2.9746022], dtype=float32)}
```

By default, we ran 8 particles for optimization, which is helpful to see that all of them found approximately the same maximum likelihood estimate.
