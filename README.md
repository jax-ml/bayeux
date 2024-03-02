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

We define a model by providing a log density in JAX. This could be defined using a probabilistic programming language (PPL) like [numpyro](examples/numpyro_and_bayeux), [PyMC](examples/pymc_and_bayeux), [TFP](examples/tfp_and_bayeux), distrax, oryx, coix, or directly in JAX.

```python
import bayeux as bx
import jax

normal_density = bx.Model(
  log_density=lambda x: -x*x,
  test_point=1.)

seed = jax.random.key(0)

opt_results = normal_density.optimize.optax_adam(seed=seed)
# OR!
idata = normal_density.mcmc.numpyro_nuts(seed=seed)
# OR!
surrogate_posterior, loss = normal_density.vi.tfp_factored_surrogate_posterior(seed=seed)
```

## Read more

* [Defining models](inference)
* [Inspecting models](inspecting)
* [Testing and debugging](debug_mode)
* Also see `bayeux` integration with [numpyro](examples/numpyro_and_bayeux), [PyMC](examples/pymc_and_bayeux), and [TFP](examples/tfp_and_bayeux)!


*This is not an officially supported Google product.*
