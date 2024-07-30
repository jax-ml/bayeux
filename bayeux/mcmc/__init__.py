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

"""Imports from submodules."""
# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
import importlib

# TFP-on-JAX always installed
from bayeux._src.mcmc.tfp import HMC as HMC_TFP
from bayeux._src.mcmc.tfp import NUTS as NUTS_TFP
from bayeux._src.mcmc.tfp import SnaperHMC as SNAPER_HMC_TFP
__all__ = ["HMC_TFP", "NUTS_TFP", "SNAPER_HMC_TFP"]

if importlib.util.find_spec("blackjax") is not None:
  from bayeux._src.mcmc.blackjax import CheesHMC as CheesHMCblackjax
  from bayeux._src.mcmc.blackjax import HMC as HMCblackjax
  from bayeux._src.mcmc.blackjax import HMCPathfinder as HMC_Pathfinder_blackjax
  from bayeux._src.mcmc.blackjax import MeadsHMC as MeadsHMCblackjax
  from bayeux._src.mcmc.blackjax import NUTS as NUTSblackjax
  from bayeux._src.mcmc.blackjax import NUTSPathfinder as NUTS_Pathfinder_blackjax
  __all__.extend(["HMCblackjax", "CheesHMCblackjax", "MeadsHMCblackjax",
                  "NUTSblackjax", "HMC_Pathfinder_blackjax",
                  "NUTS_Pathfinder_blackjax"])

if importlib.util.find_spec("flowMC") is not None:
  from bayeux._src.mcmc.flowmc import MaskedCouplingRQSplineHMC as MaskedCouplingRQSplineHMCflowmc
  from bayeux._src.mcmc.flowmc import MaskedCouplingRQSplineMALA as MaskedCouplingRQSplineMALAflowmc
  from bayeux._src.mcmc.flowmc import RealNVPHMC as RealNVPHMCflowmc
  from bayeux._src.mcmc.flowmc import RealNVPMALA as RealNVPMALAflowmc

  __all__.extend([
      "MaskedCouplingRQSplineHMCflowmc",
      "MaskedCouplingRQSplineMALAflowmc",
      "RealNVPHMCflowmc",
      "RealNVPMALAflowmc"])

if importlib.util.find_spec("numpyro") is not None:
  from bayeux._src.mcmc.numpyro import HMC as HMCnumpyro
  from bayeux._src.mcmc.numpyro import NUTS as NUTSnumpyro

  __all__.extend(["HMCnumpyro", "NUTSnumpyro"])

if importlib.util.find_spec("nutpie") is not None:
  from bayeux._src.mcmc.nutpie import _NutpieSampler as NutpieSampler

  __all__.extend(["NutpieSampler"])

