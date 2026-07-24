import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from fake_useragent import UserAgent
from pdf2image import convert_from_path
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from pypdf import PdfWriter

if TYPE_CHECKING:
    from deeppresenter.slidex.models import ExportCommandRecord
from deeppresenter.utils.constants import PACKAGE_DIR, PDF_OPTIONS
from deeppresenter.utils.log import debug, error

FAKE_UA = UserAgent()

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-blink-features=AutomationControlled",
]

ANTI_DETECTION = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
    if (!window.chrome) { window.chrome = { runtime: {} }; }
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
}
"""

ASPECT_RATIOS = {
    "16:9": {"width": "1280px", "height": "720px"},
    "4:3": {"width": "960px", "height": "720px"},
    "A1": {"width": "2244px", "height": "3178px"},
    "A2": {"width": "1587px", "height": "2244px"},
    "A3": {"width": "1122px", "height": "1587px"},
    "A4": {"width": "794px", "height": "1123px"},
}

_REQUIRED_PACKAGES = ("fast-glob", "minimist", "pptxgenjs", "playwright", "sharp")
_CACHE_NODE_MODULES = Path.home() / ".cache/deeppresenter/html2pptx/node_modules"
SCRIPT_PATH = PACKAGE_DIR / "html2pptx" / "html2pptx_cli.js"
LOCAL_NM = SCRIPT_PATH.parent / "node_modules"

class PlaywrightConverter:
    _playwright: Playwright | None = None
    _browser: Browser | None = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "PlaywrightConverter":
        """Async context manager entry"""
        async with PlaywrightConverter._lock:
            if PlaywrightConverter._browser is None:
                PlaywrightConverter._playwright = await async_playwright().start()
                PlaywrightConverter._browser = (
                    await PlaywrightConverter._playwright.chromium.launch(
                        headless=True, args=LAUNCH_ARGS
                    )
                )

        self.context = await PlaywrightConverter._browser.new_context(
            user_agent=FAKE_UA.random,
            bypass_csp=True,
        )
        await self.context.add_init_script(ANTI_DETECTION)
        self.page = await self.context.new_page()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit, only close context"""
        if self.context:
            await self.context.close()
            self.context = None
            self.page = None

    @classmethod
    async def shutdown(cls) -> None:
        """Close the shared browser and Playwright driver before the event loop exits."""
        async with cls._lock:
            browser = cls._browser
            playwright = cls._playwright
            cls._browser = None
            cls._playwright = None

            if browser is not None:
                await browser.close()
            if playwright is not None:
                await playwright.stop()

    async def convert_to_pdf(
        self,
        html_files: list[str | Path],
        output_pdf: Path | str,
        aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"],
        error_sink: list[str] | None = None,
    ) -> Path:
        if isinstance(output_pdf, str):
            output_pdf = Path(output_pdf)
        pdf_files = [tempfile.mkstemp(suffix=".pdf")[1] for _ in range(len(html_files))]
        folder = output_pdf.parent / f".slide_images-pdf-{output_pdf.stem}"
        folder.mkdir(exist_ok=True, parents=True)

        page = await self.context.new_page()
        if error_sink is not None:
            page.on(
                "pageerror",
                lambda exc: error_sink.append(f"Page error: {exc}"),
            )
            page.on(
                "console",
                lambda msg: (
                    error_sink.append(f"Console error: {msg.text}")
                    if msg.type == "error"
                    else None
                ),
            )
        try:
            for html, pdf in zip(sorted(html_files), pdf_files):
                await page.goto(Path(html).resolve().as_uri(), wait_until="networkidle")
                await page.pdf(path=pdf, **PDF_OPTIONS, **ASPECT_RATIOS[aspect_ratio])
        except Exception as e:
            error(f"Failed to convert HTML to PDF: {e}")
            raise e
        finally:
            await page.close()

        with PdfWriter() as merger:
            for pdf_file in pdf_files:
                merger.append(pdf_file)

            with open(output_pdf, "wb") as f:
                merger.write(f)

        for idx, page in enumerate(convert_from_path(output_pdf, dpi=100)):
            page.save(folder / f"slide_{(idx + 1):02d}.jpg")
        debug(f"Converted PDF saved at: {output_pdf}")
        return folder


@dataclass(frozen=True)
class Html2PptxResult:
    """Captured conversion process metadata for replay and soft-mode auditing."""

    output_path: Path | None
    command: "ExportCommandRecord"
    ignored_warnings: list[str]


class Html2PptxError(RuntimeError):
    """Strict conversion failure retaining the complete command record."""

    def __init__(self, message: str, command: "ExportCommandRecord") -> None:
        super().__init__(message)
        self.command = command


def _html2pptx_version() -> str:
    package = SCRIPT_PATH.parent / "package.json"
    package_version = json.loads(package.read_text()).get("version", "unknown")
    process = __import__("subprocess").run(
        ["node", "--version"], capture_output=True, text=True, check=False
    )
    return f"html2pptx/{package_version}; node/{process.stdout.strip() or 'unknown'}"


async def convert_html_to_pptx(
    html_inputs: Path | str | Iterable[Path | str],
    output_pptx: Path | str | None = None,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
    soft_parsing: bool = False,
) -> Html2PptxResult:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"html2pptx CLI not found at {SCRIPT_PATH}")

    validate_only = output_pptx is None
    output_path = None if validate_only else Path(output_pptx)

    html_dir: Path | None = None
    html_files: list[str] = []
    if isinstance(html_inputs, (str, Path)):
        input_path = Path(html_inputs)
        if not input_path.exists():
            raise FileNotFoundError(f"HTML input does not exist: {input_path}")
        if input_path.is_dir():
            html_dir = input_path
        else:
            html_files = [str(input_path.resolve())]
    else:
        for item in html_inputs:
            item_path = Path(item)
            if not item_path.exists():
                raise FileNotFoundError(f"HTML input does not exist: {item_path}")
            if item_path.is_dir():
                if html_dir is not None or html_files:
                    raise ValueError("html_inputs cannot mix directories and files")
                html_dir = item_path
            else:
                html_files.append(str(item_path.resolve()))

    if html_dir is None and not html_files:
        raise ValueError("No HTML inputs provided")

    cmd = ["node", str(SCRIPT_PATH), "--layout", aspect_ratio]
    if html_dir is not None:
        cmd.extend(["--html_dir", str(html_dir.resolve())])
    else:
        for html_file in html_files:
            cmd.extend(["--html", html_file])

    if validate_only:
        cmd.append("--validate")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output", str(output_path)])

    if soft_parsing:
        cmd.append("--soft")

    node_modules = (
        LOCAL_NM
        if all((LOCAL_NM / package).exists() for package in _REQUIRED_PACKAGES)
        else _CACHE_NODE_MODULES
    )
    missing_packages = [
        package for package in _REQUIRED_PACKAGES if not (node_modules / package).exists()
    ]
    if missing_packages:
        raise RuntimeError(
            "html2pptx Node.js dependencies are missing: "
            + ", ".join(missing_packages)
            + ". Run `deeppresenter onboard` first."
        )
    process_env = os.environ.copy()
    existing_node_path = process_env.get("NODE_PATH")
    process_env["NODE_PATH"] = os.pathsep.join(
        filter(None, (str(node_modules), existing_node_path))
    )

    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(SCRIPT_PATH.parent),
        env=process_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    from deeppresenter.slidex.models import ExportCommandRecord

    command = ExportCommandRecord(
        executable=cmd[0],
        arguments=cmd[1:],
        version=_html2pptx_version(),
        return_code=process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    ignored_warnings = [
        line.strip()
        for line in (stdout + "\n" + stderr).splitlines()
        if "[SOFT MODE]" in line
        or line.strip().startswith(tuple(f"{index}." for index in range(1, 100)))
    ]
    if process.returncode != 0:
        details = (stderr or stdout).strip()
        if "Cannot find module" in details:
            details = "html2pptx Node dependencies are missing"
        raise Html2PptxError(
            f"html2pptx failed: {details.split('at html2pptx (')[0]}", command
        )
    return Html2PptxResult(
        output_path=output_path,
        command=command,
        ignored_warnings=ignored_warnings,
    )


@asynccontextmanager
async def playwright_lifespan(server: Any) -> AsyncIterator[dict[str, Any]]:
    """FastMCP lifespan hook that closes the shared browser on server shutdown."""
    try:
        yield {}
    finally:
        await PlaywrightConverter.shutdown()
