"""Phase 12 performance, safety, and environment contract tests."""

from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from deeppresenter.slidex.cache import ContentCache
from deeppresenter.slidex.environment import EpisodeState, SlidexEnvironment, StepResult
from deeppresenter.slidex.inspectors.neural import AtomicNeuralClient
from deeppresenter.slidex.models import DefectClass
from deeppresenter.slidex.performance import PerformanceTracker
from deeppresenter.tools.filesystem import WorkspaceTools
from deeppresenter.utils.mineru_api import _extract_zip_bytes
from deeppresenter.utils.typings import InputRequest, sanitize_attachment_name

pytestmark = pytest.mark.unit


class CountingLLM:
    is_multimodal = False
    identifier = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._endpoints = [
            SimpleNamespace(base_url="fake", model="fake", sampling_parameters={})
        ]

    def require_capabilities(self, *capabilities: str) -> None:
        return None

    async def run(self, *args, **kwargs):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        content = json.dumps(
            {"verdict": "pass", "severity": 0, "confidence": 1, "evidence": ["clear"]}
        )
        return ChatCompletion.model_validate(
            {
                "id": "fake",
                "created": 0,
                "model": "fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_environment_state_duplicate_and_concurrent_step_guard() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def reset(seed):
        return {"seed": seed}

    async def transition(observation, action):
        started.set()
        await release.wait()
        return StepResult(observation={"value": action["value"]}, reward=1, done=False)

    environment = SlidexEnvironment(reset, transition, max_steps=2)
    await environment.reset(7)
    first = asyncio.create_task(environment.step({"action_id": "a", "value": 1}))
    await started.wait()
    with pytest.raises(RuntimeError, match="concurrent"):
        await environment.step({"action_id": "b", "value": 2})
    release.set()
    assert not (await first).done
    with pytest.raises(ValueError, match="duplicate"):
        await environment.step({"action_id": "a", "value": 1})
    result = await environment.step({"action_id": "b", "value": 2})
    assert result.done and result.info["termination"] == "max_steps"
    assert environment.state is EpisodeState.TERMINATED
    with pytest.raises(RuntimeError, match="active"):
        await environment.step({"action_id": "c", "value": 3})


def test_workspace_rejects_escape_and_bounds_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tools = WorkspaceTools(workspace, max_output_bytes=8)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace / "link").symlink_to(outside)
    for path in ("../outside.txt", str(outside), "link"):
        with pytest.raises(ValueError, match="escapes"):
            tools.read_file(path)
    result = json.loads(tools.run_command("printf 1234567890"))
    assert result["stdout"] == "12345678"
    assert result["stdout_truncated"]
    assert not result["stderr_truncated"]
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        tools.run_command("sleep 10 & wait", timeout=0.05)
    assert time.monotonic() - started < 2


def test_attachment_names_and_archive_extraction_are_safe(tmp_path: Path) -> None:
    assert sanitize_attachment_name("safe file.pdf") == "safe file.pdf"
    for name in ("../bad", ".hidden", "..", "a/b"):
        with pytest.raises(ValueError):
            sanitize_attachment_name(name)

    source = tmp_path / "normal.txt"
    source.write_text("ok")
    request = InputRequest(instruction="x", attachments=[str(source)])
    request.copy_to_workspace(tmp_path / "workspace")
    assert Path(request.attachments[0]).name == "normal.txt"

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="escapes"):
        _extract_zip_bytes(payload.getvalue(), str(tmp_path / "extract"))
    assert not (tmp_path / "escape.txt").exists()

    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.txt", b"0" * 100_000)
    with pytest.raises(ValueError, match="compression ratio"):
        _extract_zip_bytes(
            bomb.getvalue(), str(tmp_path / "bomb"), max_compression_ratio=2
        )


@pytest.mark.asyncio
async def test_neural_cache_and_concurrency_limit() -> None:
    model = CountingLLM()
    cached = AtomicNeuralClient(model, cache_results=True)
    first = await cached.inspect(DefectClass.S1, "definition", {"artifact_hash": "a"})
    second = await cached.inspect(DefectClass.S1, "definition", {"artifact_hash": "a"})
    assert first == second and model.calls == 1

    model = CountingLLM()
    limited = AtomicNeuralClient(model, max_concurrent=2)
    await asyncio.gather(
        *(
            limited.inspect(DefectClass.S1, "definition", {"artifact_hash": str(index)})
            for index in range(6)
        )
    )
    assert model.peak == 2


def test_content_cache_and_performance_summary(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path / "cache")
    key = cache.key("inspection", {"artifact": "a"}, "G1", "router-v1")
    cache.put_json("inspection", key, {"status": "pass"})
    assert cache.get_json("inspection", key) == {"status": "pass"}

    tracker = PerformanceTracker()
    for latency in (1, 2, 3, 4, 100):
        tracker.record("critic", latency, episode_id="episode", cost=0.1)
    summary = tracker.summary()
    assert summary["operations"]["critic"] == {
        "count": 5,
        "p50_latency_ms": 3,
        "p95_latency_ms": 100,
    }
    assert summary["episode_costs"]["episode"] == pytest.approx(0.5)
