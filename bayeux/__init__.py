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

"""bayeux API."""

# A new PyPI release will be pushed everytime `__version__` is increased
# When changing this, also update the CHANGELOG.md
__version__ = '0.1.15'

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570
# pylint: disable=useless-import-alias
from bayeux import mcmc as mcmc
from bayeux import optimize as optimize
from bayeux._src import debug as debug
from bayeux._src import initialization as initialization
from bayeux._src import shared as shared
from bayeux._src import types as types
from bayeux._src.bayeux import Model  as Model  # pylint: disable=g-importing-member
