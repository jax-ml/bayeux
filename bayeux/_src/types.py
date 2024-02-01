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

"""Type aliases for library."""
from typing import Any, Callable, Mapping, NamedTuple, Union
import jax.numpy as jnp

Point = Union[Mapping[str, jnp.ndarray], NamedTuple]
JAXFn = Callable[[Point], Point]
MaybeJAXFn = Union[JAXFn, Any]
LogDensityFn = Callable[[Point], jnp.ndarray]
