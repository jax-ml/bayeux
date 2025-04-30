# Inspecting models

## Seeing keyword arguments

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

```python
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

## Available algorithms

Algorithms are sometimes dynamically determined at runtime, based on the libraries that are installed ("pay for what you need"). A model can give programmatic access to the available algorithms via `methods`:

```python
normal_model.methods

{'mcmc': ['tfp_hmc',
  'tfp_nuts',
  'tfp_snaper_hmc',
  'blackjax_hmc',
  'blackjax_chees_hmc',
  'blackjax_meads_hmc',
  'blackjax_nuts',
  'blackjax_hmc_pathfinder',
  'blackjax_nuts_pathfinder',
  'numpyro_hmc',
  'numpyro_nuts'],
 'optimize': ['jaxopt_bfgs',
  'jaxopt_gradient_descent',
  'jaxopt_lbfgs',
  'jaxopt_nonlinear_cg',
  'optimistix_bfgs',
  'optimistix_chord',
  'optimistix_dogleg',
  'optimistix_gauss_newton',
  'optimistix_indirect_levenberg_marquardt',
  'optimistix_levenberg_marquardt',
  'optimistix_nelder_mead',
  'optimistix_newton',
  'optimistix_nonlinear_cg',
  'optax_adabelief',
  'optax_adafactor',
  'optax_adagrad',
  'optax_adam',
  'optax_adamw',
  'optax_adamax',
  'optax_amsgrad',
  'optax_fromage',
  'optax_lamb',
  'optax_lion',
  'optax_noisy_sgd',
  'optax_novograd',
  'optax_radam',
  'optax_rmsprop',
  'optax_sgd',
  'optax_sm3',
  'optax_yogi'],
 'vi': ['tfp_factored_surrogate_posterior']}
```

The string representation of a model will tell you what methods are available.

```python
print(normal_density)

mcmc
    .tfp_hmc
    .tfp_nuts
    .tfp_snaper_hmc
    .blackjax_hmc
    .blackjax_chees_hmc
    .blackjax_meads_hmc
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
    .optimistix_bfgs
    .optimistix_chord
    .optimistix_dogleg
    .optimistix_gauss_newton
    .optimistix_indirect_levenberg_marquardt
    .optimistix_levenberg_marquardt
    .optimistix_nelder_mead
    .optimistix_newton
    .optimistix_nonlinear_cg
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

Note that this also works on the namespaces:

```
normal_model.optimize.methods

['jaxopt_bfgs',
 'jaxopt_gradient_descent',
 'jaxopt_lbfgs',
 'jaxopt_nonlinear_cg',
 'optimistix_bfgs',
 'optimistix_chord',
 'optimistix_dogleg',
 'optimistix_gauss_newton',
 'optimistix_indirect_levenberg_marquardt',
 'optimistix_levenberg_marquardt',
 'optimistix_nelder_mead',
 'optimistix_newton',
 'optimistix_nonlinear_cg',
 'optax_adabelief',
 'optax_adafactor',
 'optax_adagrad',
 'optax_adam',
 'optax_adamw',
 'optax_adamax',
 'optax_amsgrad',
 'optax_fromage',
 'optax_lamb',
 'optax_lion',
 'optax_noisy_sgd',
 'optax_novograd',
 'optax_radam',
 'optax_rmsprop',
 'optax_sgd',
 'optax_sm3',
 'optax_yogi']
```

and

```
print(normal_model.mcmc)

tfp_hmc
tfp_nuts
tfp_snaper_hmc
blackjax_hmc
blackjax_chees_hmc
blackjax_meads_hmc
blackjax_nuts
blackjax_hmc_pathfinder
blackjax_nuts_pathfinder
numpyro_hmc
numpyro_nuts
```
