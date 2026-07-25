"""Concurrency-safe episode state machine shared by local RL integrations."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EpisodeState(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    TERMINATED = "terminated"


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: dict[str, Any] = Field(default_factory=dict)
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


Transition = Callable[[dict[str, Any], dict[str, Any]], Awaitable[StepResult]]
Reset = Callable[[int | None], Awaitable[dict[str, Any]]]


class SlidexEnvironment:
    """Reject concurrent, duplicate, out-of-order, and post-terminal actions."""

    def __init__(
        self, reset_handler: Reset, transition: Transition, *, max_steps: int = 20
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._reset_handler = reset_handler
        self._transition = transition
        self._max_steps = max_steps
        self._lock = asyncio.Lock()
        self._state = EpisodeState.IDLE
        self._observation: dict[str, Any] = {}
        self._step_index = 0
        self._action_ids: set[str] = set()

    @property
    def state(self) -> EpisodeState:
        return self._state

    async def reset(self, seed: int | None = None) -> dict[str, Any]:
        if self._lock.locked():
            raise RuntimeError("environment operation already in progress")
        async with self._lock:
            self._observation = await self._reset_handler(seed)
            self._step_index = 0
            self._action_ids.clear()
            self._state = EpisodeState.ACTIVE
            return dict(self._observation)

    async def step(self, action: dict[str, Any]) -> StepResult:
        if self._lock.locked():
            raise RuntimeError("concurrent step is forbidden")
        async with self._lock:
            if self._state is not EpisodeState.ACTIVE:
                raise RuntimeError(f"step requires active episode, got {self._state}")
            action_id = self._action_id(action)
            if action_id in self._action_ids:
                raise ValueError(f"duplicate action: {action_id}")
            self._action_ids.add(action_id)
            result = await self._transition(dict(self._observation), dict(action))
            self._step_index += 1
            forced_done = self._step_index >= self._max_steps
            self._observation = dict(result.observation)
            if result.done or forced_done:
                self._state = EpisodeState.TERMINATED
            if forced_done and not result.done:
                result = result.model_copy(
                    update={
                        "done": True,
                        "info": {**result.info, "termination": "max_steps"},
                    }
                )
            return result

    @staticmethod
    def _action_id(action: dict[str, Any]) -> str:
        explicit = action.get("action_id")
        if explicit:
            return str(explicit)
        payload = json.dumps(
            action, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()
