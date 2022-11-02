# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Substrate builder."""

from collections.abc import Mapping, Sequence
from typing import Any

import chex
import dm_env
import rx
from rx import subject

from meltingpot.python.utils.substrates.wrappers import base


@chex.dataclass(frozen=True)
class SubstrateObservables:
  """Observables for a substrate.

  Attributes:
    action: emits actions sent to the substrate from players.
    timestep: emits timesteps sent from the substrate to players.
    events: emits environment-specific events resulting from any interactions
      with the Substrate. Each individual event is emitted as a single element:
      (event_name, event_item).
  """
  action: rx.typing.Observable[Sequence[int]]
  timestep: rx.typing.Observable[dm_env.TimeStep]
  events: rx.typing.Observable[tuple[str, Any]]


class Substrate(base.Lab2dWrapper):
  """Specific subclass of Wrapper with overridden spec types."""

  def __init__(self, env: ...) -> None:
    """See base class."""
    super().__init__(env)
    self._action_subject = subject.Subject()
    self._timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    self._observables = SubstrateObservables(
        action=self._action_subject,
        events=self._events_subject,
        timestep=self._timestep_subject,
    )

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def step(self, action: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    self._action_subject.on_next(action)
    timestep = super().step(action)
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    return self._env.reward_spec()

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    return self._env.observation_spec()

  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    return self._env.action_spec()

  def close(self) -> None:
    """See base class."""
    super().close()
    self._action_subject.on_completed()
    self._timestep_subject.on_completed()
    self._events_subject.on_completed()

  def observables(self) -> SubstrateObservables:
    """Returns observables for the substrate."""
    return self._observables
