# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/jax-ml/bayeux/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [0.1.13] - 2024-08-13

* Prepare for more blackjax API changes.

## [0.1.13] - 2024-07-10

* Prepare for blackjax API change.

## [0.1.12] - 2024-05-23

* Add schedule free Adam optimizer
* Replace jax.tree_* with jax.tree.*
* Remove optimistix fixed point finders from API

## [0.1.11] - 2024-04-08

### Update for new version of flowMC
### Unpin scipy
### Add pymc as dependency

## [0.1.10] - 2024-04-03

### Keep Python 3.9 compatibility.

## [0.1.9] - 2024-02-27

### Add programmatic access to algorithms

## [0.1.8] - 2024-02-14

### Add HMC and NUTS from TFP
### Small change to blackjax default step size

## [0.1.7] - 2024-02-13

### Add SNAPER HMC from TFP
### Fix flowMC keyword handling

## [0.1.6] - 2024-02-01

### Add samplers from flowMC

## [0.1.5] - 2024-01-12

### Bugfix for PyMC models

## [0.1.4] - 2024-01-11

### Allow automatic model creation from PyMC

## [0.1.3] - 2024-01-10

### Allow automatic model creation from numpyro and TFP
### Add optimistix support
### Add meads and chees from blackjax
### Add documentation

## [0.1.2] - 2024-01-03

### Slightly improved Python 3.9 support

## [0.1.1] - 2023-12-20

### Initial release

[Unreleased]: https://github.com/jax-ml/bayeux/compare/v0.1.10...HEAD
[0.1.10]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.10
[0.1.9]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.9
[0.1.8]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.8
[0.1.7]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.7
[0.1.6]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.6
[0.1.5]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.5
[0.1.4]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.4
[0.1.3]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.3
[0.1.2]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.2
[0.1.1]: https://github.com/jax-ml/bayeux/releases/tag/v0.1.1
