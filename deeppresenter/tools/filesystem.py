"""Workspace-scoped filesystem and command tools for local agents."""

import fnmatch
import json
import os
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag


# File extensions where a double-escaped tool-call payload (the model emitted
# a JSON string whose *value* is itself JSON-escaped, e.g. a literal ``\\n``
# and ``\\"`` instead of a real newline/quote) breaks the underlying markup.
# Some small LLMs occasionally escape multi-line HTML/CSS content twice when
# filling a JSON tool-call argument. The corrupted file still "looks like"
# valid content to `edit_file`'s exact-string matching (old/new both contain
# the same literal escapes), so the model can loop indefinitely believing its
# edits landed while the browser-rendered layout never actually changes.
_ESCAPE_CHECKED_SUFFIXES = (".html", ".css")
_MIN_ESCAPE_HITS_TO_FLAG = 3


def _looks_double_escaped(content: str) -> bool:
    """Detect literal backslash-escape sequences left over from double JSON encoding.

    Heuristic: markup-like content with several literal two-character
    backslash-n / backslash-quote sequences is almost certainly the result
    of encoding the JSON string value twice, not intentional text. HTML/CSS
    never legitimately contains a run of literal backslash-quote sequences
    (attribute quoting always uses a real quote character), so counting
    them is reliable even when an edit_file call only corrupts part of an
    otherwise normal file that still has some real newlines elsewhere.
    """
    escaped_newlines = content.count(chr(92) + "n")
    escaped_quotes = content.count(chr(92) + chr(34))
    return (escaped_newlines + escaped_quotes) >= _MIN_ESCAPE_HITS_TO_FLAG


# Small models (observed with gpt-4o-mini) sometimes get stuck in a loop
# where, after being told their content is double-escaped, they re-escape
# the *already double-escaped* string instead of fixing it -- turning \n
# into \\n on the next retry. Telling the model to "rewrite from scratch"
# does not reliably break this loop. Since the escaping is a deterministic,
# reversible transformation, we auto-correct it instead of just rejecting:
# the unescaped content is unambiguously what the model meant to write.
_MAX_UNESCAPE_PASSES = 4

_UNESCAPE_MAP = {
    chr(92) + "n": "\n",
    chr(92) + "t": "\t",
    chr(92) + "r": "\r",
    chr(92) + chr(34): chr(34),
    chr(92) + "'": "'",
    chr(92) * 2: chr(92),
}


def _unescape_once(content: str) -> str:
    r"""Undo one layer of literal backslash-escaping (\n -> newline, etc.).

    Processed as a single left-to-right scan so a literal ``\\n`` (escaped
    backslash followed by a literal ``n``) is not mis-parsed as ``\n``.
    """
    result: list[str] = []
    i = 0
    length = len(content)
    while i < length:
        char = content[i]
        if char == chr(92) and i + 1 < length:
            pair = content[i : i + 2]
            if pair == chr(92) * 2:
                result.append(chr(92))
                i += 2
                continue
            replacement = _UNESCAPE_MAP.get(pair)
            if replacement is not None:
                result.append(replacement)
                i += 2
                continue
        result.append(char)
        i += 1
    return "".join(result)


def _fix_double_escaping(target_path: Path, content: str) -> str:
    """Return content with double-escaping auto-corrected, or raise if unsafe.

    Repeatedly unescapes until the heuristic no longer flags the content
    (handles the observed case of escaping compounding across retries) or
    until ``_MAX_UNESCAPE_PASSES`` is reached without resolving, at which
    point we give up and reject rather than risk corrupting real content.
    """
    if target_path.suffix.lower() not in _ESCAPE_CHECKED_SUFFIXES:
        return content
    if not _looks_double_escaped(content):
        return content
    fixed = content
    for _ in range(_MAX_UNESCAPE_PASSES):
        fixed = _unescape_once(fixed)
        if not _looks_double_escaped(fixed):
            return fixed
    raise ValueError(
        f"Refusing to write {target_path.name}: content still looks "
        "double-escaped after automatic unescaping (literal backslash-n / "
        "backslash-quote sequences instead of real newlines and quotes). "
        "Pass the raw HTML/CSS text as the JSON string value directly -- do "
        "not JSON-encode it a second time before putting it in the "
        "`content`/`new` argument."
    )


def _resolve_edit_match(
    content: str, old: str, new: str
) -> tuple[str, str, int]:
    """Find a unique ``old`` occurrence, falling back to an unescaped ``old``.

    Small models occasionally track file content in "escaped" form (as if
    still inside a JSON string literal) after a prior ``write_file`` call
    auto-corrected double-escaping on disk. Their next ``edit_file`` then
    passes a literal ``\\"``/``\\n``-laden ``old`` that can never match the
    already-clean file, causing the exact same failing call to repeat
    indefinitely. If the exact match fails, retry once against an unescaped
    ``old`` (and correspondingly unescape ``new``, so the replacement text
    does not reintroduce the same escaping into the file).
    """
    matches = content.count(old)
    if matches == 1:
        return old, new, matches
    # Unlike _looks_double_escaped's >=3 heuristic (tuned for whole-file
    # content), an `old` search snippet can be short enough to contain just
    # one or two literal escape sequences yet still be the double-escaped
    # cause of a 0-match failure. Any escape sequence at all is worth a
    # fallback retry here because a successful retry still requires the
    # unescaped string to match uniquely -- an unrelated `old` won't match.
    has_any_escape = (chr(92) + "n") in old or (chr(92) + chr(34)) in old
    if not has_any_escape:
        return old, new, matches
    unescaped_old = old
    unescaped_new = new
    for _ in range(_MAX_UNESCAPE_PASSES):
        unescaped_old = _unescape_once(unescaped_old)
        unescaped_new = _unescape_once(unescaped_new)
        if not _looks_double_escaped(unescaped_old):
            break
    fallback_matches = content.count(unescaped_old)
    if fallback_matches == 1:
        return unescaped_old, unescaped_new, fallback_matches
    return old, new, matches


class WorkspaceTools:
    """Tools for a trusted local agent, not an isolation or multi-tenant boundary."""

    def __init__(self, workspace: Path, *, max_output_bytes: int = 1_000_000):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes

    def _resolve(self, path: str = ".") -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute() and candidate.parts[:2] == ("/", "workspace"):
            candidate = self.workspace.joinpath(*candidate.parts[2:])
        elif not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        """Read UTF-8 text, optionally selecting a zero-based line range."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        lines = (
            self._resolve(path).read_text(encoding="utf-8").splitlines(keepends=True)
        )
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        return "".join(selected)

    def write_file(self, path: str, content: str) -> str:
        """Write a UTF-8 text file inside the workspace, creating parent directories."""
        target = self._resolve(path)
        content = _fix_double_escaping(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def edit_file(
        self,
        path: str | None = None,
        old: str | None = None,
        new: str | None = None,
        html_file: str | None = None,
    ) -> str:
        """Replace one unique occurrence; ``html_file`` aliases ``path``."""
        target_path = path or html_file
        if target_path is None:
            raise ValueError("path or html_file is required")
        if path is not None and html_file is not None and path != html_file:
            raise ValueError("path and html_file refer to different files")
        if old is None or new is None:
            raise ValueError("old and new are required")
        target = self._resolve(target_path)
        content = target.read_text(encoding="utf-8")
        old, new, matches = _resolve_edit_match(content, old, new)
        if matches != 1:
            raise ValueError(
                f"Expected exactly one match in {target_path}, found {matches}"
            )
        updated = content.replace(old, new, 1)
        updated = _fix_double_escaping(target, updated)
        target.write_text(updated, encoding="utf-8")
        return str(target)

    _PATCHABLE_STYLE_PROPERTIES = frozenset(
        {
            "align-items", "background", "background-color", "border", "border-color",
            "border-radius", "bottom", "color", "column-gap", "display", "flex",
            "flex-basis", "flex-direction", "flex-grow", "flex-shrink", "flex-wrap",
            "font-family", "font-size", "font-style", "font-weight", "gap",
            "grid-template-columns", "grid-template-rows", "height", "justify-content",
            "left", "letter-spacing", "line-height", "margin", "margin-bottom",
            "margin-left", "margin-right", "margin-top", "max-height", "max-width",
            "min-height", "min-width", "object-fit", "opacity", "overflow", "padding",
            "padding-bottom", "padding-left", "padding-right", "padding-top", "position",
            "right", "row-gap", "text-align", "top", "transform", "vertical-align",
            "visibility", "white-space", "width", "z-index",
        }
    )
    _PATCHABLE_ATTRIBUTES = frozenset({"alt", "aria-label", "href", "src", "title"})

    @staticmethod
    def _inline_styles(element: Tag) -> dict[str, str]:
        return {
            key.strip().lower(): item.strip()
            for declaration in str(element.get("style", "")).split(";")
            if ":" in declaration
            for key, item in [declaration.split(":", 1)]
            if key.strip()
        }

    @staticmethod
    def _stable_id_index(soup: BeautifulSoup) -> dict[str, list[Tag]]:
        index: dict[str, list[Tag]] = {}
        for element in soup.select("[data-slidex-id]"):
            stable_id = str(element.get("data-slidex-id", "")).strip()
            if stable_id:
                index.setdefault(stable_id, []).append(element)
        return index

    def _find_stable_element(
        self, soup: BeautifulSoup, path: str, element_id: str
    ) -> tuple[Tag, dict[str, list[Tag]]]:
        normalized_id = element_id.strip()
        if not normalized_id:
            raise ValueError("element_id must be a non-empty data-slidex-id")
        index = self._stable_id_index(soup)
        matches = index.get(normalized_id, [])
        if len(matches) == 1:
            return matches[0], index
        candidates = sorted(index)[:80]
        if not matches:
            raise ValueError(
                f"No element with data-slidex-id={normalized_id!r} in {path}. "
                f"Available IDs: {candidates}. Call inspect_slide_element(path) "
                "to inspect the current source index; do not invent an ID."
            )
        raise ValueError(
            f"data-slidex-id={normalized_id!r} is duplicated in {path}; "
            "repair duplicate IDs before applying a targeted patch"
        )

    def inspect_slide_element(self, path: str, element_id: str | None = None) -> str:
        """Inspect source-local HTML by stable ID before a repair.

        With ``element_id``, return the exact element's text, inline styles,
        allowed attributes, parent ID, and descendant IDs. Without it, return a
        compact index of every ``data-slidex-id`` element. This is source
        inspection only; use ``inspect_slide`` afterwards for rendered verdicts.
        """
        target = self._resolve(path)
        if target.suffix.lower() != ".html":
            raise ValueError("inspect_slide_element only accepts .html files")
        soup = BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
        index = self._stable_id_index(soup)
        if element_id is None:
            elements = []
            for stable_id, matches in sorted(index.items()):
                element = matches[0]
                elements.append(
                    {
                        "element_id": stable_id,
                        "tag": element.name,
                        "text": element.get_text(" ", strip=True)[:240],
                        "duplicate": len(matches) > 1,
                    }
                )
            return json.dumps(
                {"path": str(target), "element_count": len(elements), "elements": elements},
                ensure_ascii=False,
            )

        element, _ = self._find_stable_element(soup, path, element_id)
        parent = element.parent if isinstance(element.parent, Tag) else None
        parent_id = str(parent.get("data-slidex-id")) if parent and parent.has_attr("data-slidex-id") else None
        children = [
            {
                "element_id": str(child.get("data-slidex-id")) if child.has_attr("data-slidex-id") else None,
                "tag": child.name,
                "text": child.get_text(" ", strip=True)[:160],
            }
            for child in element.find_all(recursive=False)
            if isinstance(child, Tag)
        ]
        attributes = {
            name: str(element.get(name))
            for name in self._PATCHABLE_ATTRIBUTES
            if element.has_attr(name)
        }
        return json.dumps(
            {
                "path": str(target),
                "element_id": element_id,
                "tag": element.name,
                "text": element.get_text(" ", strip=True),
                "inline_style": self._inline_styles(element),
                "classes": list(element.get("class", [])),
                "attributes": attributes,
                "parent_element_id": parent_id,
                "children": children,
                "descendant_element_ids": [
                    str(item.get("data-slidex-id"))
                    for item in element.select("[data-slidex-id]")
                ],
            },
            ensure_ascii=False,
        )

    def patch_slide_element(
        self,
        path: str,
        element_id: str,
        styles: dict[str, str] | None = None,
        text: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> str:
        """Patch exactly one stable-ID element with validated, auditable changes.

        This is the preferred HTML repair tool. ``styles`` permits only safe
        layout/typography properties; ``attributes`` permits alt/aria-label/href/
        src/title. ``text`` is allowed only on a leaf element, so a repair cannot
        silently delete nested content or stable IDs. Re-run ``inspect_slide``
        after every successful patch.
        """
        if not any(value is not None for value in (styles, text, attributes)):
            raise ValueError("provide at least one of styles, text, or attributes")
        target = self._resolve(path)
        if target.suffix.lower() != ".html":
            raise ValueError("patch_slide_element only accepts .html files")
        soup = BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
        element, _ = self._find_stable_element(soup, path, element_id)
        before = {
            "inline_style": self._inline_styles(element),
            "text": element.get_text(" ", strip=True),
            "attributes": {name: str(element.get(name)) for name in self._PATCHABLE_ATTRIBUTES if element.has_attr(name)},
        }
        changed: dict[str, Any] = {}
        if styles is not None:
            invalid = sorted(set(styles) - self._PATCHABLE_STYLE_PROPERTIES)
            if invalid:
                raise ValueError(f"unsupported style properties: {invalid}")
            if any(not isinstance(value, str) or not value.strip() for value in styles.values()):
                raise ValueError("style values must be non-empty strings")
            merged = self._inline_styles(element)
            merged.update({name.lower(): value.strip() for name, value in styles.items()})
            element["style"] = "; ".join(f"{name}: {value}" for name, value in merged.items())
            changed["styles"] = {name.lower(): value.strip() for name, value in styles.items()}
        if attributes is not None:
            invalid = sorted(set(attributes) - self._PATCHABLE_ATTRIBUTES)
            if invalid:
                raise ValueError(f"unsupported attributes: {invalid}")
            if any(not isinstance(value, str) for value in attributes.values()):
                raise ValueError("attribute values must be strings")
            for name, value in attributes.items():
                element[name] = value
            changed["attributes"] = attributes
        if text is not None:
            if element.find(True) is not None:
                leaf_ids = [
                    str(descendant.get("data-slidex-id"))
                    for descendant in element.select("[data-slidex-id]")
                    if descendant.find(True) is None
                ]
                raise ValueError(
                    f"data-slidex-id={element_id!r} is a container and cannot receive a "
                    "text patch because nested content and IDs must be preserved. Use styles "
                    "on this container, or patch one of its leaf child IDs with text instead: "
                    f"{leaf_ids}. Do not retry this container text patch."
                )
            element.string = text
            changed["text"] = text
        target.write_text(str(soup), encoding="utf-8")
        after = {
            "inline_style": self._inline_styles(element),
            "text": element.get_text(" ", strip=True),
            "attributes": {name: str(element.get(name)) for name in self._PATCHABLE_ATTRIBUTES if element.has_attr(name)},
        }
        return json.dumps(
            {
                "path": str(target), "element_id": element_id, "changed": changed,
                "before": before, "after": after,
            },
            ensure_ascii=False,
        )

    def patch_html(
        self,
        path: str,
        selector: str,
        operation: Literal["set_style", "remove_style", "set_attribute", "remove_attribute", "add_class", "remove_class", "replace_text"] = "set_style",
        name: str | None = None,
        value: str | None = None,
    ) -> str:
        """Apply one deterministic DOM patch to exactly one HTML element.

        Use ``[data-slidex-id="..."]`` for visible content and a CSS selector
        such as ``body`` or ``.slide-content`` for shared layout containers.
        This avoids brittle exact-string replacement when fixing inspected HTML.
        """
        target = self._resolve(path)
        if target.suffix.lower() != ".html":
            raise ValueError("patch_html only accepts .html files")
        soup = BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
        matches = soup.select(selector)
        if len(matches) != 1:
            raise ValueError(f"selector must match exactly one element in {path}, found {len(matches)}")
        element = matches[0]
        if operation in {"set_style", "remove_style"}:
            if not name:
                raise ValueError("style operation requires name")
            styles = {
                key.strip().lower(): item.strip()
                for declaration in str(element.get("style", "")).split(";")
                if ":" in declaration
                for key, item in [declaration.split(":", 1)]
            }
            if operation == "set_style":
                if value is None:
                    raise ValueError("set_style requires value")
                styles[name.strip().lower()] = value
            else:
                styles.pop(name.strip().lower(), None)
            if styles:
                element["style"] = "; ".join(f"{key}: {item}" for key, item in styles.items())
            else:
                element.attrs.pop("style", None)
        elif operation in {"set_attribute", "remove_attribute"}:
            if not name:
                raise ValueError("attribute operation requires name")
            if operation == "set_attribute":
                if value is None:
                    raise ValueError("set_attribute requires value")
                element[name] = value
            else:
                element.attrs.pop(name, None)
        elif operation in {"add_class", "remove_class"}:
            if not value:
                raise ValueError("class operation requires value")
            classes = list(element.get("class", []))
            if operation == "add_class" and value not in classes:
                classes.append(value)
            if operation == "remove_class":
                classes = [item for item in classes if item != value]
            if classes:
                element["class"] = classes
            else:
                element.attrs.pop("class", None)
        else:
            if value is None:
                raise ValueError("replace_text requires value")
            element.clear()
            element.append(value)
        target.write_text(str(soup), encoding="utf-8")
        return json.dumps(
            {"path": str(target), "selector": selector, "operation": operation, "name": name, "value": value},
            ensure_ascii=False,
        )

    def list_files(self, path: str = ".", pattern: str = "**/*") -> str:
        """List workspace files below path matching a glob pattern."""
        root = self._resolve(path)
        if not root.is_dir():
            raise NotADirectoryError(path)
        files = sorted(
            str(item.relative_to(self.workspace))
            for item in root.glob(pattern)
            if item.is_file()
        )
        return json.dumps(files, ensure_ascii=False)

    def search_files(self, query: str, path: str = ".", glob: str = "*") -> str:
        """Search text files in the workspace, preferring ripgrep when available."""
        root = self._resolve(path)
        if shutil.which("rg"):
            result = subprocess.run(
                [
                    "rg",
                    "--line-number",
                    "--color",
                    "never",
                    "--glob",
                    glob,
                    query,
                    str(root),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode not in {0, 1}:
                raise RuntimeError(result.stderr.strip() or "rg failed")
            return result.stdout

        matches: list[str] = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for file_path in candidates:
            relative = file_path.relative_to(root if root.is_dir() else root.parent)
            if not file_path.is_file() or not (
                fnmatch.fnmatch(str(relative), glob)
                or fnmatch.fnmatch(file_path.name, glob)
            ):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query in line:
                    relative = file_path.relative_to(self.workspace)
                    matches.append(f"{relative}:{line_number}:{line}")
        return "\n".join(matches) + ("\n" if matches else "")

    def run_command(self, command: str, cwd: str = ".", timeout: float = 120) -> str:
        """Run a shell command from a workspace directory and return structured output."""
        working_directory = self._resolve(cwd)
        if not working_directory.is_dir():
            raise NotADirectoryError(cwd)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError(f"Command timed out after {timeout} seconds") from exc
        stdout_text, stdout_truncated = self._bounded_output(stdout)
        stderr_text, stderr_truncated = self._bounded_output(stderr)
        payload: dict[str, object] = {
            "exit_code": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        if stdout_truncated or stderr_truncated:
            payload.update(
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
        return json.dumps(payload, ensure_ascii=False)

    def _bounded_output(self, value: bytes) -> tuple[str, bool]:
        truncated = len(value) > self.max_output_bytes
        selected = value[: self.max_output_bytes]
        return selected.decode(errors="replace"), truncated

    def register(self, agent_env: object) -> None:
        """Register all workspace tools on an AgentEnv-like registry."""
        for tool in (
            self.read_file,
            self.write_file,
            self.edit_file,
            self.patch_html,
            self.inspect_slide_element,
            self.patch_slide_element,
            self.list_files,
            self.search_files,
            self.run_command,
        ):
            agent_env.register_tool(tool)
