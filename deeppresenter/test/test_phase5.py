import hashlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from deeppresenter.slidex.attribution import FailureAttributor
from deeppresenter.slidex.inspectors.neural import (
    AtomicNeuralClient,
    DensityInspector,
    ImageTextContradictionInspector,
    NeuralCapabilityError,
    RenderAnomalyInspector,
    TitleBodyMismatchInspector,
)
from deeppresenter.slidex.inspectors.reference import ReferenceInspector
from deeppresenter.slidex.models import (
    AttributionLabel,
    BoundingBox,
    ComputedSlideElement,
    ComputedSlideIR,
    DeclaredSlideIR,
    DefectClass,
    InspectionContext,
    InspectionStatus,
    PairwiseVerdict,
    Provenance,
    SlideArtifact,
    SlideElement,
)
from deeppresenter.utils.config import LLM


class FakeLLM:
    def __init__(self, responses: list[str], *, multimodal: bool = True) -> None:
        self.is_multimodal = multimodal
        self.identifier = "fake-endpoint"
        self._endpoints = [SimpleNamespace(base_url="http://fake/v1", model="fake", sampling_parameters={"temperature": 0, "seed": 42})]
        self.responses = iter(responses)
        self.requests = []

    async def run(self, messages, response_format=None, retry_times=1):
        self.requests.append((messages, response_format, retry_times))
        raw = next(self.responses)
        return ChatCompletion.model_validate({
            "id": "fake", "created": 0, "model": "fake", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": raw}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })


def _verdict(verdict="pass", **kwargs):
    value = {"verdict": verdict, "severity": 0 if verdict != "fail" else 1, "confidence": 0.9, "evidence": ["observable fact"], "element_ids": ["body"]}
    value.update(kwargs)
    import json
    return json.dumps(value)


def _artifact(*, overflow=False, width=1280, height=720):
    title = SlideElement(element_id="title", tag="h1", semantic_role="title", text="Revenue")
    body = SlideElement(element_id="body", tag="p", semantic_role="body", text="Revenue increased")
    image = SlideElement(element_id="image", tag="img", semantic_role="image")
    computed = ComputedSlideElement(
        element_id="body", tag="p", semantic_role="body", text="Revenue increased",
        bbox={"x": 10, "y": 10, "width": 100, "height": 20, "page_width": width, "page_height": height},
        client_width=100, client_height=20, scroll_width=120 if overflow else 100, scroll_height=20,
    )
    return SlideArtifact(
        artifact_id="a", source_uri="slide.html", source_sha256=hashlib.sha256(b"x").hexdigest(),
        declared_ir=DeclaredSlideIR(slide_id="s", page_width=width, page_height=height, elements=[title, body, image]),
        computed_ir=ComputedSlideIR(slide_id="s", page_width=width, page_height=height, elements=[computed], browser="Chromium", browser_version="1"),
        provenance=Provenance(creation_action="test"),
    )


@pytest.mark.unit
def test_critic_requires_explicit_multimodal_capability():
    config = {
        "file_path": "x", "research_agent": {"model": "m", "api_key": "x"}, "design_agent": {"model": "m", "api_key": "x"},
        "long_context_model": {"model": "m", "api_key": "x"}, "critic_model": {"model": "critic", "api_key": "x"},
    }
    from deeppresenter.utils.config import DeepPresenterConfig
    with pytest.raises(ValidationError, match="explicitly declared"):
        DeepPresenterConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_atomic_s1_payload_is_single_class_structured_and_stateless(tmp_path: Path):
    image = tmp_path / "slide.png"
    image.write_bytes(b"png")
    model = FakeLLM([_verdict("fail", repair_suggestion="Align the body topic")])
    client = AtomicNeuralClient(model, require_multimodal=True)
    result = await TitleBodyMismatchInspector(client).inspect(InspectionContext(artifact=_artifact(), render_path=str(image)))
    assert result.status == InspectionStatus.FAIL
    assert result.defect_class == DefectClass.S1
    assert result.element_ids == ["body"]
    messages, schema, retries = model.requests[0]
    prompt = messages[0]["content"][0]["text"]
    assert "exactly one defect class: S1" in prompt and "mutation" not in prompt.lower().replace("mutation metadata", "")
    assert schema.__name__ == "AtomicVerdict" and retries == 1
    assert client.records[0].usage["total_tokens"] == 15


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_json_and_missing_verdict_become_error():
    for raw in ("not json", '{"severity": 0, "confidence": 1, "evidence": ["x"]}'):
        result = await TitleBodyMismatchInspector(AtomicNeuralClient(FakeLLM([raw]))).inspect(InspectionContext(artifact=_artifact()))
        assert result.status == InspectionStatus.ERROR


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_without_image_support_is_capability_error(tmp_path: Path):
    image = tmp_path / "slide.png"
    image.write_bytes(b"png")
    model = FakeLLM([_verdict()], multimodal=False)
    with pytest.raises(NeuralCapabilityError):
        AtomicNeuralClient(model, require_multimodal=True)
    result = await TitleBodyMismatchInspector(AtomicNeuralClient(model)).inspect(InspectionContext(artifact=_artifact(), render_path=str(image)))
    assert result.status == InspectionStatus.ERROR
    assert result.repair_hint.action == "configure_provider_capability"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s4_deterministic_extreme_avoids_model():
    artifact = _artifact()
    artifact.computed_ir.elements[0].text = "x" * 901
    model = FakeLLM([])
    result = await DensityInspector(AtomicNeuralClient(model)).inspect(InspectionContext(artifact=artifact))
    assert result.status == InspectionStatus.FAIL and not model.requests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s6_missing_render_defers_and_g7_dom_fail_avoids_model():
    model = FakeLLM([])
    client = AtomicNeuralClient(model)
    s6 = await ImageTextContradictionInspector(client, "image", "body").inspect(InspectionContext(artifact=_artifact()))
    g7 = await RenderAnomalyInspector(client, "body").inspect(InspectionContext(artifact=_artifact(overflow=True)))
    assert s6.status == InspectionStatus.DEFER
    assert g7.status == InspectionStatus.FAIL and not model.requests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reference_missing_defers_and_order_control(tmp_path: Path):
    target_path = tmp_path / "target.png"; target_path.write_bytes(b"a")
    ref_path = tmp_path / "ref.png"; ref_path.write_bytes(b"b")
    missing = await ReferenceInspector(AtomicNeuralClient(FakeLLM([]))).inspect(InspectionContext(artifact=_artifact()), DefectClass.S6, "contradiction")
    assert missing.status == InspectionStatus.DEFER
    responses = [
        '{"verdict":"right","confidence":0.9,"evidence":["right is cleaner"]}',
        '{"verdict":"left","confidence":0.9,"evidence":["left is cleaner"]}',
        '{"verdict":"tie","confidence":1,"evidence":["same image"]}',
    ]
    result = await ReferenceInspector(AtomicNeuralClient(FakeLLM(responses))).inspect(
        InspectionContext(artifact=_artifact(), render_path=str(target_path), reference_artifact=_artifact(), reference_render_path=str(ref_path)),
        DefectClass.S6, "contradiction",
    )
    assert result.status == InspectionStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failure_attribution_reproduces_abc_and_budget(tmp_path: Path):
    image = tmp_path / "slide.png"; image.write_bytes(b"x")
    responses = [
        _verdict("defer", defer_reason="not visible"), _verdict("fail"), _verdict("fail"), _verdict("fail"), _verdict("fail")
    ]
    model = FakeLLM(responses)
    attribution = await FailureAttributor(AtomicNeuralClient(model)).run(
        InspectionContext(artifact=_artifact(), render_path=str(image)), DefectClass.S1,
        "title-body contradiction", {"title": "A", "body": "B"},
        whole_rubric_definition="whole rubric baseline", repeated_whole_rubric_budget=2,
    )
    assert attribution.label == AttributionLabel.STRUCTURE_RESCUED
    assert set(attribution.conditions) == {"A_image", "B_structured_ir", "C_image_ir"}
    assert len(attribution.whole_rubric) == 2 and len(attribution.records) == 5


@pytest.mark.unit
def test_fake_openai_server_receives_schema_and_429_becomes_error():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append((self.path, payload))
            if len(requests) == 2:
                self.send_response(429)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit"}}')
                return
            content = _verdict("pass")
            body = json.dumps({
                "id": "fake", "created": 0, "model": "fake", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = LLM(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fake", api_key="fake",
            is_multimodal=False, sampling_parameters={"temperature": 0, "top_p": 1, "seed": 42},
            client_kwargs={"max_retries": 0},
        )
        client = AtomicNeuralClient(model)
        verdict, _ = __import__("asyncio").run(client.inspect(DefectClass.S1, "single defect", {"title": "x"}))
        assert verdict.verdict == "pass"
        assert requests[0][0] == "/v1/chat/completions"
        assert requests[0][1]["response_format"]["type"] == "json_schema"
        with pytest.raises(ValueError, match="429"):
            __import__("asyncio").run(client.inspect(DefectClass.S1, "single defect", {"title": "x"}))
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_becomes_structured_error():
    class TimeoutLLM(FakeLLM):
        async def run(self, messages, response_format=None, retry_times=1):
            raise TimeoutError("provider timeout")

    result = await TitleBodyMismatchInspector(AtomicNeuralClient(TimeoutLLM([]))).inspect(
        InspectionContext(artifact=_artifact())
    )
    assert result.status == InspectionStatus.ERROR
    assert "provider timeout" in result.evidence[0].detail

ATOMIC_PROMPT_FIXTURES = {
    DefectClass.S1: "Fail only when body claims contradict or clearly concern a different topic than the title; missing detail alone passes.",
    DefectClass.S4: "Classify only over-packed or under-packed information density relative to the page role; intentional minimal title slides pass.",
    DefectClass.S6: "For exactly one supplied image and adjacent caption or claim, fail only when visible image content contradicts that text. Defer when the image cannot establish the claim.",
    DefectClass.G7: "Determine only whether content crosses or is clipped by its specified container boundary in the rendered slide.",
}


@pytest.mark.unit
def test_human_reviewed_atomic_prompt_fixtures_remain_single_defect():
    for defect_class, definition in ATOMIC_PROMPT_FIXTURES.items():
        prompt = AtomicNeuralClient._atomic_prompt(defect_class, definition, {"element_ids": ["target"]})
        assert f"exactly one defect class: {defect_class.value}" in prompt
        assert "general quality" in prompt
        assert "ground-truth" not in prompt.lower()


@pytest.mark.llm
@pytest.mark.asyncio
async def test_live_atomic_model_from_dotenv():
    import os
    from dotenv import load_dotenv

    if os.getenv("SLIDEX_RUN_LLM_TESTS") != "1":
        pytest.skip("set SLIDEX_RUN_LLM_TESTS=1 to run credentialed model tests")
    load_dotenv()
    base_url = os.environ["SLIDER_LLM_BASE_URL"].rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
    model = LLM(
        base_url=base_url,
        model=os.environ["SLIDER_JUDGE_LLM_MODEL"],
        api_key=os.environ["SLIDER_LLM_API_KEY"],
        is_multimodal=False,
        sampling_parameters={"temperature": 0, "top_p": 1, "seed": 42},
    )
    verdict, record = await AtomicNeuralClient(model).inspect(
        DefectClass.S1,
        ATOMIC_PROMPT_FIXTURES[DefectClass.S1],
        {"titles": [{"element_id": "title", "text": "Solar growth"}], "bodies": [{"element_id": "body", "text": "Solar capacity doubled."}]},
    )
    assert verdict.verdict == "pass"
    assert record.usage
