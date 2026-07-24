"""Contract tests for the outbound OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from pydantic import BaseModel

from deeppresenter.slidex.models import PolicyCallRecord, TrajectoryStep
from deeppresenter.utils.config import LLM, ModelCapabilityError


class StructuredAnswer(BaseModel):
    answer: str


class FakeServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, bytes, str, float]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                owner.requests.append(json.loads(self.rfile.read(length)))
                status, body, content_type, delay = owner.responses.pop(0)
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.thread.join()

    def reply(self, message: dict[str, Any], *, finish_reason: str = "stop") -> None:
        body = {
            "id": "chatcmpl-local",
            "created": 1,
            "model": "local-model",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": message}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        self.responses.append((200, json.dumps(body).encode(), "application/json", 0))


def model(server: FakeServer, **kwargs: Any) -> LLM:
    config: dict[str, Any] = {
        "base_url": server.url,
        "model": "local-model",
        "api_key": "secret",
        "capabilities": {
            "text": True,
            "vision": True,
            "tools": True,
            "structured_output": True,
        },
        "client_kwargs": {"max_retries": 0},
    }
    config.update(kwargs)
    return LLM(**config)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_multimodal_and_structured_payloads() -> None:
    with FakeServer() as server:
        server.reply({"role": "assistant", "content": "hello", "reasoning": "brief"})
        client = model(server)
        response = await client.run("hi", retry_times=1)
        assert response.choices[0].message.content == "hello"
        assert client.last_call["usage"]["total_tokens"] == 7
        assert client.last_call["reasoning"] == ["brief"]

        server.reply({"role": "assistant", "content": '{"answer":"seen"}'})
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ]
        parsed = await client.run(
            messages, response_format=StructuredAnswer, retry_times=1
        )
        assert json.loads(parsed.choices[0].message.content) == {"answer": "seen"}
        assert server.requests[1]["messages"][0]["content"][1]["type"] == "image_url"
        assert server.requests[1]["response_format"]["type"] == "json_schema"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_choice_and_multiple_tool_calls_are_preserved() -> None:
    with FakeServer() as server:
        calls = [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "first", "arguments": '{"x":1}'},
            },
            {
                "id": "b",
                "type": "function",
                "function": {"name": "second", "arguments": "{}"},
            },
        ]
        server.reply(
            {"role": "assistant", "content": None, "tool_calls": calls},
            finish_reason="tool_calls",
        )
        tools = [
            {
                "type": "function",
                "function": {"name": "first", "parameters": {"type": "object"}},
            }
        ]
        client = model(server)
        response = await client.run(
            "act", tools=tools, tool_choice="required", retry_times=1
        )
        assert [
            call.function.name for call in response.choices[0].message.tool_calls
        ] == ["first", "second"]
        assert server.requests[0]["tool_choice"] == "required"
        assert client.last_call["finish_reasons"] == ["tool_calls"]
        assert len(client.last_call["tool_calls"][0]) == 2


@pytest.mark.unit
def test_capability_validation_env_secret_and_trajectory_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_KEY", "very-secret")
    client = LLM(
        model="local",
        api_key="$LOCAL_LLM_KEY",
        capabilities={"text": True, "tools": False},
    )
    assert "very-secret" not in repr(client)
    assert "very-secret" not in json.dumps(client.model_dump(mode="json"))
    with pytest.raises(ModelCapabilityError, match="tools"):
        client.require_capabilities("tools")

    record = PolicyCallRecord(
        endpoint_identifier="local",
        provider="openai",
        model="m",
        sampling_parameters={"temperature": 0},
        usage={"total_tokens": 7},
        finish_reasons=["stop"],
        reasoning=[None],
        tool_calls=[[]],
        response_hash="a" * 64,
        attempt=1,
    )
    step = TrajectoryStep(step_index=0, action={"type": "sample"}, policy_call=record)
    assert step.policy_call.sampling_parameters == {"temperature": 0}


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
async def test_http_failures_are_not_fabricated(status: int) -> None:
    with FakeServer() as server:
        body = json.dumps(
            {"error": {"message": "provider failure", "type": "api_error"}}
        ).encode()
        server.responses.append((status, body, "application/json", 0))
        with pytest.raises(ValueError, match=str(status)):
            await model(server).run("hi", retry_times=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_json_and_timeout_are_explicit_errors() -> None:
    with FakeServer() as server:
        server.responses.append((200, b"not-json", "application/json", 0))
        with pytest.raises(ValueError, match="JSON"):
            await model(server).run("hi", retry_times=1)

    with FakeServer() as server:
        server.reply({"role": "assistant", "content": "late"})
        status, body, content_type, _ = server.responses.pop()
        server.responses.append((status, body, content_type, 0.2))
        with pytest.raises(
            ValueError,
            match="timeout|timed out|Timeout",
        ):
            await model(server, client_kwargs={"timeout": 0.02, "max_retries": 0}).run(
                "hi", retry_times=1
            )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incomplete_tool_call_is_rejected() -> None:
    with FakeServer() as server:
        server.reply(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "bad", "type": "function", "function": {"name": "first"}}
                ],
            },
            finish_reason="tool_calls",
        )
        tools = [
            {
                "type": "function",
                "function": {"name": "first", "parameters": {"type": "object"}},
            }
        ]
        with pytest.raises(ValueError):
            await model(server).run("act", tools=tools, retry_times=1)
