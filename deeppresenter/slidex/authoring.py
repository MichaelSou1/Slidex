"""Versioned operating contract for agents that author Slidex HTML slides."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag

AUTHORING_SKILL_VERSION = "1.3"
_CANVAS = {
    "16:9": (1280, 720),
    "4:3": (960, 720),
    "A1": (2244, 3178),
    "A2": (1587, 2244),
    "A3": (1122, 1587),
    "A4": (794, 1123),
}


@dataclass(frozen=True)
class PreflightFinding:
    """One deterministic authoring-contract violation."""

    code: str
    message: str
    element_ids: tuple[str, ...] = ()
    severity: Literal["error", "warning"] = "error"
    suggested_operation: str = "policy_edit"

    def model_dump(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "element_ids": list(self.element_ids),
            "severity": self.severity,
            "suggested_operation": self.suggested_operation,
        }


def authoring_skill() -> str:
    """Return the short, shared agent-harness operating contract."""
    return """<slidex_authoring_skill version=\"1.2\">
Use this protocol for every Slidex deck. First draft every required slide and only then spend remaining budget on repairs. Do not spend more than one preflight/inspect cycle on a slide before all required slides exist.

HTML canvas contract: use a fixed body matching the requested canvas; set `box-sizing: border-box` on `html, body, *`; do not add padding to a fixed-size body. Put the 24px safe margin as padding on an inner `.slide-content` container; do not use outer margins on the first/last flow child because CSS margin collapse can make the body scroll. Keep every visible element inside that container. Give each independently visible content element a stable, unique `data-slidex-id`. In particular, every visible list item (`li`), card, table row/cell, and process step needs its own ID; never put an ID only on its `ul`, grid, or other ancestor.

For each slide: write -> call `preflight_slide` -> fix every returned error -> call `inspect_slide`. In a repair-only run, the prompt supplies the exact current stable-ID index for every slide: use only those IDs, never invent semantic IDs. The index marks `text_patchable`; only `true` leaf IDs may receive a text patch, while container IDs are style-only. First call `inspect_slide_element` for the exact target ID, then use `patch_slide_element` as the default: it deterministically changes one ID-addressed element's approved inline styles, leaf text, or safe attributes and reports the before/after state. Use `patch_html` only for a shared layout container (`body` or `.slide-content`) or class-level changes. Use `edit_file` only as a low-frequency escape hatch after reading the file; never guess an `edit_file` old string. If preflight says `fixed_canvas_padding`, move padding to the inner container; if it says `canvas_size_mismatch`, repair root dimensions; if it says `missing_stable_id` or `duplicate_stable_id`, fix IDs without renumbering existing ones. Never solve overflow by hiding, deleting required content, moving it off-canvas, or changing body to max-width/max-height.

Use `inspect_slide` reports only when this arm exposes feedback. In repair-only runs, source mutation tools and `inspect_slide` must be in separate turns: make one targeted patch, then your next turn must call `inspect_slide` for that slide before any more source patch. Use the fresh hard findings and proposed repair actions as the sole repair acceptance criterion. When `repair_strategy` says `NO_IMPROVEMENT`, stop repeating micro-style patches on the listed IDs: choose a different action/target or structural layout strategy, or leave it unresolved and continue. If repair budget is exhausted, leave the finding unresolved and continue to the next slide. Finish by calling `finalize`.
</slidex_authoring_skill>"""


def stable_id_inventory(html_path: Path) -> dict[str, Any]:
    """Return the compact, current repair target index for one HTML slide.

    The index is deliberately source-derived at repair start rather than inferred
    from critic prose: model repair calls must address IDs that actually exist in
    the cloned workspace.  It remains a hint, so the agent still uses
    ``inspect_slide_element`` before patching an individual node.
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    elements: list[dict[str, str]] = []
    for node in soup.select("[data-slidex-id]"):
        element_id = str(node.get("data-slidex-id", "")).strip()
        if not element_id:
            continue
        elements.append(
            {
                "element_id": element_id,
                "tag": str(node.name),
                "text": node.get_text(" ", strip=True)[:160],
                "text_patchable": node.find(True) is None,
            }
        )
    return {"slide": html_path.name, "elements": elements}


def authoring_skill_hash() -> str:
    """Return a stable hash recorded with each agent run."""
    return hashlib.sha256(authoring_skill().encode("utf-8")).hexdigest()


def preflight_html(
    html_path: Path, aspect_ratio: str = "16:9"
) -> dict[str, Any]:
    """Check deterministic source-level mistakes before expensive rendering."""
    if aspect_ratio not in _CANVAS:
        raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")
    markup = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(markup, "lxml")
    body = soup.body
    findings: list[PreflightFinding] = []
    if body is None:
        findings.append(PreflightFinding("missing_body", "HTML must contain a body element."))
    else:
        style = _style_map(str(body.get("style", "")))
        width, height = _CANVAS[aspect_ratio]
        if not _matches_px(style.get("width"), width) or not _matches_px(style.get("height"), height):
            findings.append(
                PreflightFinding(
                    "canvas_size_mismatch",
                    f"body must use width:{width}px and height:{height}px for {aspect_ratio}.",
                    suggested_operation="resize_container",
                )
            )
        if _nonzero_length(style.get("padding")) or any(
            _nonzero_length(style.get(key))
            for key in ("padding-top", "padding-right", "padding-bottom", "padding-left")
        ):
            findings.append(
                PreflightFinding(
                    "fixed_canvas_padding",
                    "A fixed-size body cannot carry padding; move safe margins to an inner .slide-content container.",
                    suggested_operation="resize_container",
                )
            )
        if _nonzero_length(style.get("margin")) or any(
            _nonzero_length(style.get(key))
            for key in ("margin-top", "margin-right", "margin-bottom", "margin-left")
        ):
            findings.append(
                PreflightFinding(
                    "fixed_canvas_margin",
                    "A fixed-size body must not carry outer margins; set body margin to 0 and place spacing in .slide-content.",
                    suggested_operation="resize_container",
                )
            )
        if "border-box" not in style.get("box-sizing", ""):
            findings.append(
                PreflightFinding(
                    "missing_border_box",
                    "Set box-sizing:border-box on html, body, and descendants.",
                    severity="warning",
                    suggested_operation="policy_edit",
                )
            )
    visible = [node for node in soup.select("p, h1, h2, h3, h4, li, img, table, figure, svg") if isinstance(node, Tag)]
    ids = [str(node.get("data-slidex-id", "")).strip() for node in visible]
    missing = tuple(str(node.name) for node, element_id in zip(visible, ids, strict=True) if not element_id)
    if missing:
        findings.append(PreflightFinding("missing_stable_id", "Inspectable visible elements need non-empty data-slidex-id values.", missing))
    duplicates = tuple(sorted({element_id for element_id in ids if element_id and ids.count(element_id) > 1}))
    if duplicates:
        findings.append(PreflightFinding("duplicate_stable_id", "data-slidex-id values must be unique per slide.", duplicates))
    return {
        "skill_version": AUTHORING_SKILL_VERSION,
        "skill_hash": authoring_skill_hash(),
        "html_file": str(html_path),
        "ok": not any(item.severity == "error" for item in findings),
        "findings": [item.model_dump() for item in findings],
    }


def _style_map(style: str) -> dict[str, str]:
    return {
        key.strip().lower(): value.strip().lower()
        for declaration in style.split(";")
        if ":" in declaration
        for key, value in [declaration.split(":", 1)]
    }


def _matches_px(value: str | None, expected: int) -> bool:
    return value is not None and re.fullmatch(rf"{expected}(?:\.0+)?px", value.strip()) is not None


def _nonzero_length(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"0", "0px", "0.0px", "initial", "unset"}
