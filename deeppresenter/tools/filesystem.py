"""Workspace-scoped filesystem and command tools for local agents."""

import fnmatch
import json
import shutil
import subprocess
from pathlib import Path


class WorkspaceTools:
    """Expose predictable local tools constrained to one workspace."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

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

    def read_file(
        self, path: str, offset: int = 0, limit: int | None = None
    ) -> str:
        """Read UTF-8 text, optionally selecting a zero-based line range."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        lines = self._resolve(path).read_text(encoding="utf-8").splitlines(
            keepends=True
        )
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        return "".join(selected)

    def write_file(self, path: str, content: str) -> str:
        """Write a UTF-8 text file inside the workspace, creating parent directories."""
        target = self._resolve(path)
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
        matches = content.count(old)
        if matches != 1:
            raise ValueError(
                f"Expected exactly one match in {target_path}, found {matches}"
            )
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return str(target)

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
        try:
            result = subprocess.run(
                command,
                cwd=working_directory,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command timed out after {timeout} seconds") from exc
        return json.dumps(
            {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
        )

    def register(self, agent_env: object) -> None:
        """Register all workspace tools on an AgentEnv-like registry."""
        for tool in (
            self.read_file,
            self.write_file,
            self.edit_file,
            self.list_files,
            self.search_files,
            self.run_command,
        ):
            agent_env.register_tool(tool)
