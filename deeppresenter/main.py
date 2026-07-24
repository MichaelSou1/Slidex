import hashlib
import json
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from deeppresenter.agents.design import Design
from deeppresenter.agents.env import AgentEnv
from deeppresenter.agents.planner import Planner
from deeppresenter.agents.pptagent import PPTAgent
from deeppresenter.agents.research import Research
from deeppresenter.agents.subagent import SubAgent
from deeppresenter.slidex.artifacts import ArtifactStore
from deeppresenter.slidex.browser import BrowserObserver, extract_declared_ir
from deeppresenter.slidex.critic import HybridCritic, persist_report
from deeppresenter.slidex.deck import DeckInspector, enforce_export_gate
from deeppresenter.slidex.export import (
    FinalExportService,
    RenderFidelityValidator,
    extract_pptx_structure,
    pptx_to_slide_artifacts,
)
from deeppresenter.slidex.grounding import GroundingEvaluator
from deeppresenter.slidex.models import (
    InspectionContext,
    FinalArtifactStatus,
    RepairAction,
    Provenance,
    RenderArtifact,
    RendererInfo,
    SlideArtifact,
)
from deeppresenter.slidex.reward import (
    EfficiencyUsage,
    RewardConfig,
    RewardEngine,
    build_task_outcome,
)
from deeppresenter.slidex.repair import (
    DeterministicRepairer,
    actions_from_report,
    append_repair_trajectory,
    bind_after_artifact,
    compare_reports,
    detect_policy_violations,
)
from deeppresenter.tools.filesystem import WorkspaceTools
from deeppresenter.utils.config import DeepPresenterConfig
from deeppresenter.utils.constants import WORKSPACE_BASE
from deeppresenter.utils.log import debug, error, set_logger, timer
from deeppresenter.utils.outline import Outline
from deeppresenter.utils.typings import ChatMessage, ConvertType, InputRequest, Role
from deeppresenter.utils.webview import convert_html_to_pptx
from pypdf import PdfReader


def _exact_page_count(value: str | None) -> int | None:
    if not value or not value.strip().isdigit():
        return None
    return int(value.strip())


def _outline_titles(path: str | Path | None) -> list[str]:
    if not path:
        return []
    outline_path = Path(path)
    if not outline_path.is_file():
        return []
    outline = Outline.model_validate_json(outline_path.read_text(encoding="utf-8"))
    return [slide.title for slide in outline.slides]


def _required_terms(request: InputRequest) -> list[str]:
    value = request.extra_info.get("required_terms", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _grounding_sources(attachments: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    readable = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml"}
    for item in attachments:
        path = Path(item)
        if not path.is_file():
            continue
        if path.suffix.lower() in readable:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue
        sources[path.resolve().as_uri()] = text
    return sources


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)


def _manifest_render_paths(manifest) -> list[Path]:
    return [
        _file_uri_path(reference.uri)
        for name, reference in sorted(manifest.output_files.items())
        if name.startswith("pptx_render_slide_")
    ]


def _efficiency_usage(agents, *, repair_steps: int, latency_ms: float) -> EfficiencyUsage:
    messages = [message for agent in agents for message in agent.chat_history]
    return EfficiencyUsage(
        tokens=sum(agent.cost.total for agent in agents),
        model_calls=sum(message.role == Role.ASSISTANT for message in messages),
        tool_calls=sum(len(message.tool_calls or []) for message in messages),
        repair_steps=repair_steps,
        latency_ms=latency_ms,
    )


class AgentLoop:
    def __init__(
        self,
        config: DeepPresenterConfig,
        session_id: str | None = None,
        workspace: Path | None = None,
        language: Literal["zh", "en"] = "en",
    ):
        self.config = config
        self.language = language
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        self.workspace = workspace or WORKSPACE_BASE / session_id
        self.intermediate_output: dict[str, str | Path] = {}
        self.agent = None
        set_logger(
            f"deeppresenter-loop-{self.workspace.stem}",
            self.workspace / ".history" / "deeppresenter-loop.log",
        )
        debug(f"Initialized AgentLoop with workspace={self.workspace}")
        debug(f"Config: {self.config.model_dump_json(indent=2)}")

    @timer("DeepPresenter Loop")
    async def run(
        self,
        request: InputRequest,
        check_llms: bool = False,
        soft_parsing: bool = False,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """Main loop for DeepPresenter generation process.
        Arguments:
            request: InputRequest object containing task details.
            check_llms: Whether to check LLM availability before running.
            soft_parsing: Explicitly enable warning-ignoring html2pptx soft mode.
        Yields:
            ChatMessage or final output path (str). Outline path stored in intermediate_output["outline"].
        """
        run_started = time.perf_counter()
        participating_agents = []
        if not self.config.design_agent.is_multimodal and self.config.heavy_reflect:
            debug(
                "Reflective design requires a multimodal LLM in the design agent, reflection will only enable on textual state."
            )
        if check_llms:
            await self.config.validate_llms()
        request.copy_to_workspace(self.workspace)
        with open(self.workspace / ".input_request.json", "w") as f:
            json.dump(request.model_dump(), f, ensure_ascii=False, indent=2)
        async with AgentEnv(self.workspace, self.config) as agent_env:
            WorkspaceTools(self.workspace).register(agent_env)

            def thinking(thought: str) -> str:
                """Record an explicit reasoning checkpoint before the next action."""
                return thought

            agent_env.register_tool(thinking)
            agent_env.register_tool(
                SubAgent.delegate(self.config, agent_env, self.workspace, self.language)
            )
            if "deeppresenter" in agent_env._server_tools:
                agent_env._server_tools["deeppresenter"] = [
                    tool
                    for tool in agent_env._server_tools["deeppresenter"]
                    if tool != "inspect_slide"
                ]
                agent_env._tool_to_server.pop("inspect_slide", None)
                agent_env._tools_dict.pop("inspect_slide", None)

            inspected_artifacts: dict[str, SlideArtifact] = {}
            inspection_reports = {}
            inspection_rounds: dict[str, int] = {}
            pending_repairs: dict[str, list[RepairAction]] = {}
            repair_proposals: dict[str, RepairAction] = {}
            latest_artifact_ids: dict[str, str] = {}
            latest_source_hashes: dict[str, str] = {}

            async def inspect_slide(
                html_file: str,
                aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
            ) -> str:
                """Inspect one revision; every source repair requires this fresh check."""
                html_path = WorkspaceTools(self.workspace)._resolve(html_file)
                key = str(html_path)
                rounds = inspection_rounds.get(key, 0)
                if rounds >= self.config.slidex.max_repair_rounds + 1:
                    prior = inspection_reports.get(key)
                    if prior is None:
                        raise RuntimeError(
                            "repair budget exhausted before initial inspection"
                        )
                    payload = prior.model_dump()
                    payload["summary"]["terminal_reason"] = "max_repair_rounds"
                    payload["summary"]["repair_rounds"] = rounds - 1
                    return json.dumps(
                        payload, ensure_ascii=False, indent=2, default=str
                    )
                # Syntax/conversion failures are preflight errors, not quality repair
                # rounds. Charge the budget only after a report can be produced.
                await convert_html_to_pptx(html_path, aspect_ratio=aspect_ratio)
                observation_dir = (
                    self.workspace / ".history" / "observations" / html_path.stem
                )
                observation = await BrowserObserver().observe(
                    html_path,
                    observation_dir,
                    slide_id=html_path.stem,
                    debug_overlay=True,
                )
                declared = extract_declared_ir(
                    html_path,
                    slide_id=html_path.stem,
                    global_css=html_path.parent / "global.css"
                    if (html_path.parent / "global.css").exists()
                    else None,
                )
                declared_path = observation_dir / "declared_ir.json"
                computed_path = observation_dir / "computed_ir.json"
                declared_path.write_text(declared.model_dump_json(indent=2))
                computed_path.write_text(
                    observation.computed_ir.model_dump_json(indent=2)
                )
                if not observation.computed_ir.render_ready:
                    raise RuntimeError(
                        f"Slide render is not ready: {observation.computed_ir.resource_errors + observation.computed_ir.page_errors}"
                    )

                store = ArtifactStore(
                    self.workspace / ".history" / "slidex",
                    max_workspace_bytes=self.config.slidex.max_workspace_bytes,
                    max_artifacts=self.config.slidex.max_artifacts_per_episode,
                )
                episode_id = getattr(inspect_slide, "episode_id", None)
                if episode_id is None:
                    episode = store.create_episode(
                        versions={
                            "taxonomy": self.config.slidex.taxonomy_version,
                            "router": self.config.slidex.router_version,
                        }
                    )
                    episode_id = episode.episode_id
                    inspect_slide.episode_id = episode_id
                source = html_path.read_bytes()
                renderer = RendererInfo(
                    name=observation.computed_ir.browser,
                    version=observation.computed_ir.browser_version,
                    options={
                        "viewport": [
                            observation.computed_ir.page_width,
                            observation.computed_ir.page_height,
                        ]
                    },
                )
                provenance = Provenance(
                    parent_artifact_id=latest_artifact_ids.get(key),
                    creation_action="inspect_slide",
                    versions={"taxonomy": self.config.slidex.taxonomy_version},
                )
                slide_artifact = SlideArtifact(
                    artifact_id="pending",
                    source_uri=f"source/{html_path.name}",
                    source_sha256=hashlib.sha256(source).hexdigest(),
                    declared_ir=declared,
                    computed_ir=observation.computed_ir,
                    renders=[
                        RenderArtifact(
                            kind="html",
                            uri="renders/render.png",
                            sha256=hashlib.sha256(
                                observation.screenshot_path.read_bytes()
                            ).hexdigest(),
                            width=int(observation.computed_ir.page_width),
                            height=int(observation.computed_ir.page_height),
                            renderer=renderer,
                        )
                    ],
                    provenance=provenance,
                )
                files: dict[str, Path] = {
                    f"source/{html_path.name}": html_path,
                    "ir/declared.json": declared_path,
                    "ir/computed.json": computed_path,
                    "renders/render.png": observation.screenshot_path,
                    "renders/render.pdf": observation.pdf_path,
                }
                if observation.overlay_path:
                    files["renders/overlay.png"] = observation.overlay_path
                manifest = store.write_artifact(
                    episode_id, files, provenance, slide_artifact
                )
                persisted_artifact = slide_artifact.model_copy(
                    update={"artifact_id": manifest.artifact_id}
                )
                previous_artifact_id = latest_artifact_ids.get(key)
                previous_source_hash = latest_source_hashes.get(key)
                latest_artifact_ids[key] = manifest.artifact_id
                latest_source_hashes[key] = slide_artifact.source_sha256
                critic = HybridCritic(
                    self.config.slidex,
                    critic_model=self.config.critic_model,
                    semantic_model=self.config.semantic_model,
                )
                report = await critic.inspect(
                    InspectionContext(
                        artifact=persisted_artifact,
                        render_path=str(observation.screenshot_path),
                    )
                )
                violations = detect_policy_violations(persisted_artifact)
                report = report.model_copy(update={"policy_violations": violations})
                report_uri = persist_report(
                    store,
                    episode_id,
                    report,
                    parent_artifact_id=manifest.artifact_id,
                )
                report = report.model_copy(update={"report_uri": report_uri})
                explicit_repairs = pending_repairs.pop(key, [])
                previous_report = inspection_reports.get(key)
                for pending in explicit_repairs:
                    append_repair_trajectory(
                        self.workspace / ".history" / "slidex" / "repair_actions.jsonl",
                        bind_after_artifact(
                            pending,
                            manifest.artifact_id,
                            before_report=previous_report,
                            after_report=report,
                        ),
                    )
                if (
                    previous_artifact_id
                    and previous_source_hash != slide_artifact.source_sha256
                    and not explicit_repairs
                ):
                    source_ids = (
                        [
                            action.source_inspection_ids[0]
                            for action in actions_from_report(previous_report)
                        ]
                        if previous_report
                        else []
                    ) or ["inspection-unattributed"]
                    inferred = RepairAction(
                        action_id=f"repair-{uuid.uuid4().hex[:12]}",
                        operation="policy_edit",
                        target_ids=sorted(
                            {
                                element_id
                                for result in previous_report.results
                                for element_id in result.element_ids
                            }
                        )
                        or ["slide-root"],
                        constraints={"source_sha256": slide_artifact.source_sha256},
                        source_inspection_ids=source_ids,
                        before_artifact_id=previous_artifact_id,
                        after_artifact_id=manifest.artifact_id,
                        status="applied",
                        policy_reason="Source changed outside deterministic repair tooling.",
                        defect_delta=(
                            compare_reports(previous_report, report)
                            if previous_report
                            else []
                        ),
                    )
                    append_repair_trajectory(
                        self.workspace / ".history" / "slidex" / "repair_actions.jsonl",
                        inferred,
                    )
                inspected_artifacts[html_path.stem] = persisted_artifact
                inspection_reports[key] = report
                inspection_rounds[key] = rounds + 1
                failed_results = [
                    finding
                    for finding in report.results
                    if finding.status.value in {"fail", "error"}
                ]
                proposed_actions = actions_from_report(report)
                repair_proposals.update(
                    {action.action_id: action for action in proposed_actions}
                )
                payload = {
                    "artifact_id": report.artifact_id,
                    "slide_id": report.slide_id,
                    "summary": dict(report.summary),
                    "findings": [
                        {
                            "defect_class": finding.defect_class.value,
                            "status": finding.status.value,
                            "severity": finding.severity,
                            "element_ids": finding.element_ids,
                            "evidence": [item.detail for item in finding.evidence],
                            "repair_hint": finding.repair_hint.model_dump()
                            if hasattr(finding.repair_hint, "model_dump")
                            else finding.repair_hint,
                        }
                        for finding in failed_results
                    ],
                    "repair_actions": [
                        action.model_dump() for action in proposed_actions
                    ],
                    "report_uri": report.report_uri,
                }
                payload["summary"]["repair_rounds"] = rounds + 1
                payload["summary"]["repair_rounds_remaining"] = max(
                    0, self.config.slidex.max_repair_rounds - rounds
                )
                payload["summary"]["hard_policy_violations"] = len(violations)
                return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

            def apply_repair(html_file: str, action: dict) -> str:
                """Apply one proposed low-risk RepairAction; re-run inspect_slide next."""
                html_path = WorkspaceTools(self.workspace)._resolve(html_file)
                current = inspected_artifacts.get(html_path.stem)
                report = inspection_reports.get(str(html_path))
                if current is None or report is None:
                    raise ValueError("inspect the slide before applying a repair")

                action_id = action.get("action_id")
                proposal = repair_proposals.get(action_id)
                payload = proposal.model_dump() if proposal is not None else {}
                payload.update(action)
                repair = RepairAction.model_validate(payload)
                if repair.before_artifact_id != current.artifact_id:
                    raise ValueError(
                        "repair action does not target the latest inspected artifact"
                    )
                executed = DeterministicRepairer().apply(html_path, repair)
                append_repair_trajectory(
                    self.workspace / ".history" / "slidex" / "repair_actions.jsonl",
                    executed,
                )
                if executed.status.value == "applied":
                    pending_repairs.setdefault(str(html_path), []).append(executed)
                return executed.model_dump_json(indent=2)

            async def render_slide(
                html_file: str,
                aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
            ) -> str:
                """Render a visual preview without making a quality judgment."""
                html_path = WorkspaceTools(self.workspace)._resolve(html_file)
                output_dir = self.workspace / ".history" / "previews" / html_path.stem
                observation = await BrowserObserver().observe(
                    html_path,
                    output_dir,
                    slide_id=html_path.stem,
                    debug_overlay=False,
                )
                return str(observation.screenshot_path)

            agent_env.register_tool(inspect_slide)
            agent_env.register_tool(apply_repair)
            agent_env.register_tool(render_slide)
            hello_message = f"DeepPresenter running in {self.workspace}, with {len(request.attachments)} attachments, prompt={request.instruction}"
            modes = []
            if self.config.offline_mode:
                modes.append("Offline Mode")
            self.agent_env = agent_env
            if self.config.multiagent_mode:
                modes.append("Multiagent Mode")
            if modes:
                hello_message += f" [{', '.join(modes)}]"
            debug(hello_message)

            yield ChatMessage(role=Role.SYSTEM, content=hello_message)

            # ── Optional Planner phase ────────────────────────────────────
            if request.enable_planner:
                self.planner = Planner(
                    self.config,
                    agent_env,
                    self.workspace,
                    self.language,
                    max_turns=self.config.slidex.max_episode_steps,
                )
                self.agent = self.planner
                participating_agents.append(self.planner)
                self.planner_gen = self.planner.loop(request)
                try:
                    async for msg in self.planner_gen:
                        if isinstance(msg, str):
                            self.intermediate_output["outline"] = msg
                            yield msg
                            break
                        yield msg
                except Exception as e:
                    error_message = f"Planner agent failed with error: {e}\n{traceback.format_exc()}"
                    error(error_message)
                    raise e
                finally:
                    self.planner.save_history()
                    await self.planner_gen.aclose()
                    self.save_results()

            self.research_agent = Research(
                self.config,
                agent_env,
                self.workspace,
                self.language,
                max_turns=self.config.slidex.max_episode_steps,
            )
            self.agent = self.research_agent
            participating_agents.append(self.research_agent)
            try:
                async for msg in self.research_agent.loop(
                    request, self.intermediate_output.get("outline", None)
                ):
                    if isinstance(msg, str):
                        md_file = Path(msg)
                        if not md_file.is_absolute():
                            md_file = self.workspace / md_file
                        self.intermediate_output["manuscript"] = md_file
                        msg = str(md_file)
                        break
                    yield msg
            except Exception as e:
                error_message = (
                    f"Research agent failed with error: {e}\n{traceback.format_exc()}"
                )
                error(error_message)
                raise e
            finally:
                self.research_agent.save_history()
                self.save_results()

            if request.convert_type == ConvertType.PPTAGENT:
                self.pptagent = PPTAgent(
                    self.config,
                    agent_env,
                    self.workspace,
                    self.language,
                    max_turns=self.config.slidex.max_episode_steps,
                )
                self.agent = self.pptagent
                participating_agents.append(self.pptagent)
                try:
                    async for msg in self.pptagent.loop(request, str(md_file)):
                        if isinstance(msg, str):
                            pptx_file = Path(msg)
                            if not pptx_file.is_absolute():
                                pptx_file = self.workspace / pptx_file
                            self.intermediate_output["pptx"] = pptx_file
                            self.intermediate_output["final"] = pptx_file
                            msg = str(pptx_file)
                            break
                        yield msg
                except Exception as e:
                    error_message = (
                        f"PPTAgent failed with error: {e}\n{traceback.format_exc()}"
                    )
                    error(error_message)
                    raise e
                finally:
                    self.pptagent.save_history()
                    self.save_results()
                export_service = FinalExportService()
                export_manifest = await export_service.validate_pptx(
                    pptx_file,
                    expected_page_count=_exact_page_count(request.num_pages),
                )
                export_manifest_path = (
                    self.workspace / ".history" / "slidex" / "export_manifest.json"
                )
                export_service.save_manifest(export_manifest, export_manifest_path)
                self.intermediate_output["artifact_manifest"] = export_manifest_path
                self.intermediate_output["export_status"] = export_manifest.status.value
                if export_manifest.status != FinalArtifactStatus.PPTX_RENDER_VALIDATED:
                    raise RuntimeError(
                        "final PPTX validation failed: "
                        + (export_manifest.failure_reason or export_manifest.status.value)
                    )
                render_paths = _manifest_render_paths(export_manifest)
                inspected_artifacts = {
                    artifact.declared_ir.slide_id: artifact
                    for artifact in pptx_to_slide_artifacts(
                        pptx_file, render_paths, export_manifest.fidelity_report.renderer
                    )
                }
                deck_report = await DeckInspector(
                    HybridCritic(
                        self.config.slidex,
                        critic_model=self.config.critic_model,
                        semantic_model=self.config.semantic_model,
                    ),
                    self.config.slidex,
                ).inspect(
                    list(inspected_artifacts.values()),
                    approved_outline=_outline_titles(
                        self.intermediate_output.get("outline")
                    ),
                    task=request.instruction,
                )
                deck_report_path = (
                    self.workspace / ".history" / "slidex" / "deck_report.json"
                )
                deck_report_path.parent.mkdir(parents=True, exist_ok=True)
                deck_report_path.write_text(deck_report.model_dump_json(indent=2))
                self.intermediate_output["deck_inspection"] = deck_report_path
                enforce_export_gate(deck_report)
            else:
                self.designagent = Design(
                    self.config,
                    agent_env,
                    self.workspace,
                    self.language,
                    max_turns=self.config.slidex.max_episode_steps,
                )
                self.agent = self.designagent
                participating_agents.append(self.designagent)
                try:
                    async for msg in self.designagent.loop(request, str(md_file)):
                        if isinstance(msg, str):
                            slide_html_dir = Path(msg)
                            if not slide_html_dir.is_absolute():
                                slide_html_dir = self.workspace / slide_html_dir
                            self.intermediate_output["slide_html_dir"] = slide_html_dir
                            break
                        yield msg
                except Exception as e:
                    error_message = (
                        f"Design agent failed with error: {e}\n{traceback.format_exc()}"
                    )
                    error(error_message)
                    raise e
                finally:
                    self.designagent.save_history()
                    self.save_results()
                missing_inspections = {
                    path.stem for path in slide_html_dir.glob("slide_*.html")
                } - inspected_artifacts.keys()
                if missing_inspections:
                    raise RuntimeError(
                        "export blocked: slides were not inspected: "
                        + ", ".join(sorted(missing_inspections))
                    )
                deck_report = await DeckInspector(
                    HybridCritic(
                        self.config.slidex,
                        critic_model=self.config.critic_model,
                        semantic_model=self.config.semantic_model,
                    ),
                    self.config.slidex,
                ).inspect(
                    list(inspected_artifacts.values()),
                    approved_outline=_outline_titles(
                        self.intermediate_output.get("outline")
                    ),
                    task=request.instruction,
                )
                deck_report_path = (
                    self.workspace / ".history" / "slidex" / "deck_report.json"
                )
                deck_report_path.parent.mkdir(parents=True, exist_ok=True)
                deck_report_path.write_text(deck_report.model_dump_json(indent=2))
                self.intermediate_output["deck_inspection"] = deck_report_path
                enforce_export_gate(deck_report)
                pptx_path = self.workspace / f"{md_file.stem}.pptx"
                source_paths = sorted(slide_html_dir.glob("slide_*.html"))
                html_renders = [
                    self.workspace
                    / ".history"
                    / "observations"
                    / source.stem
                    / "render.png"
                    for source in source_paths
                ]
                export_service = FinalExportService(
                    validator=RenderFidelityValidator(
                        max_pixel_difference=self.config.slidex.export_max_pixel_difference,
                        min_perceptual_similarity=self.config.slidex.export_min_perceptual_similarity,
                        min_text_presence=self.config.slidex.export_min_text_presence,
                        zero_signal_threshold=self.config.slidex.mutation_zero_signal_threshold,
                    )
                )
                export_manifest = await export_service.export(
                    source_paths,
                    pptx_path,
                    html_renders,
                    aspect_ratio=request.powerpoint_type.value,
                    soft_mode=soft_parsing,
                    soft_mode_explicit=soft_parsing,
                    source_artifact_ids=[
                        inspected_artifacts[source.stem].artifact_id
                        for source in source_paths
                    ],
                    critic_report_uris=[deck_report_path.resolve().as_uri()],
                )
                export_manifest_path = (
                    self.workspace / ".history" / "slidex" / "export_manifest.json"
                )
                export_service.save_manifest(export_manifest, export_manifest_path)
                self.intermediate_output["artifact_manifest"] = export_manifest_path
                self.intermediate_output["export_status"] = export_manifest.status.value
                if export_manifest.status != FinalArtifactStatus.PPTX_RENDER_VALIDATED:
                    raise RuntimeError(
                        "final PPTX validation failed: "
                        + (
                            export_manifest.failure_reason
                            or export_manifest.status.value
                        )
                    )
                self.intermediate_output["pptx"] = str(pptx_path)
                self.intermediate_output["final"] = str(pptx_path)
                msg = pptx_path

            final_pptx = Path(self.intermediate_output["pptx"])
            slide_text, _, _, _ = extract_pptx_structure(
                final_pptx, len(inspected_artifacts)
            )
            task_outcome = build_task_outcome(
                instruction=request.instruction,
                requested_pages=request.num_pages,
                actual_page_count=len(slide_text),
                slide_text=slide_text,
                outline_titles=_outline_titles(
                    self.intermediate_output.get("outline")
                ),
                required_terms=_required_terms(request),
                language=self.language,
            )
            grounding_report = GroundingEvaluator().evaluate(
                slide_text, _grounding_sources(request.attachments)
            )
            evaluation_dir = self.workspace / ".history" / "slidex"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            task_path = evaluation_dir / "task_outcome.json"
            task_path.write_text(task_outcome.model_dump_json(indent=2))
            grounding_path = evaluation_dir / "grounding_report.json"
            grounding_path.write_text(grounding_report.model_dump_json(indent=2))
            usage = _efficiency_usage(
                participating_agents,
                repair_steps=sum(max(0, value - 1) for value in inspection_rounds.values()),
                latency_ms=(time.perf_counter() - run_started) * 1000,
            )
            reward = RewardEngine(
                RewardConfig.from_slidex_config(self.config.slidex)
            ).compute(
                list(deck_report.page_reports.values()),
                artifact_ids=[item.artifact_id for item in inspected_artifacts.values()],
                export=export_manifest,
                task=task_outcome,
                grounding=grounding_report,
                usage=usage,
                policy_violations=deck_report.policy_violations,
            )
            reward_path = evaluation_dir / "reward.json"
            reward_path.write_text(reward.model_dump_json(indent=2))
            efficiency_path = evaluation_dir / "efficiency.json"
            efficiency_path.write_text(usage.model_dump_json(indent=2))
            self.intermediate_output.update(
                {
                    "task_outcome": task_path,
                    "grounding_report": grounding_path,
                    "reward": reward_path,
                    "efficiency": efficiency_path,
                }
            )
            self.save_results()
            debug(f"DeepPresenter finished, final output at: {msg}")
            yield msg

    def save_results(self):
        with open(self.workspace / "intermediate_output.json", "w") as f:
            json.dump(
                {k: str(v) for k, v in self.intermediate_output.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
