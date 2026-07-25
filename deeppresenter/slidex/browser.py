"""Trusted source and browser observation for Slidex slide artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from deeppresenter.slidex.cache import ContentCache
from deeppresenter.slidex.models import (
    BoundingBox,
    ComputedSlideElement,
    ComputedSlideIR,
    DeclaredSlideIR,
    ObservedBoundingBox,
    SlideElement,
)

_INSPECTABLE = "[data-slidex-id], h1, h2, h3, p, li, img, svg, table, section, article"
_REMOTE_URL = re.compile(r"(?:https?:)?//", re.IGNORECASE)
_CSS_VAR = re.compile(r"(--[\w-]+)\s*:\s*([^;}]+)")
_CSS_COLOR = re.compile(r"(?:#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)|hsla?\([^)]*\))")
_CSS_FONT_SIZE = re.compile(r"font-size\s*:\s*([^;}]+)", re.IGNORECASE)
_OBSERVE_SCRIPT = r"""
(selector) => {
  const pageWidth = window.innerWidth;
  const pageHeight = window.innerHeight;
  const candidates = [...document.querySelectorAll(selector)];
  const pathOf = (node) => {
    const parts = [];
    while (node && node.nodeType === Node.ELEMENT_NODE) {
      const siblings = [...node.parentElement?.children || []].filter(x => x.tagName === node.tagName);
      parts.unshift(`${node.tagName.toLowerCase()}[${Math.max(1, siblings.indexOf(node) + 1)}]`);
      node = node.parentElement;
    }
    return parts.join('/');
  };
  const hash = (text) => {
    let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
    for (let i = 0; i < text.length; i++) {
      const ch = text.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, '0');
  };
  const selected = new Set(candidates);
  const idOf = (el) => el.dataset.slidexId?.trim() || `auto-${hash(pathOf(el)).slice(0, 16)}`;
  const box = (rect) => ({x: rect.x, y: rect.y, width: rect.width, height: rect.height,
    page_width: pageWidth, page_height: pageHeight});
  const intersection = (rect) => {
    const left = Math.max(0, rect.left), top = Math.max(0, rect.top);
    const right = Math.min(pageWidth, rect.right), bottom = Math.min(pageHeight, rect.bottom);
    return right > left && bottom > top
      ? {x: left, y: top, width: right-left, height: bottom-top,
         page_width: pageWidth, page_height: pageHeight} : null;
  };
  return candidates.map((el, index) => {
    const cs = getComputedStyle(el), rect = el.getBoundingClientRect();
    const textRects = [];
    for (const node of el.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE || !node.textContent.trim()) continue;
      const range = document.createRange(); range.selectNodeContents(node);
      for (const r of range.getClientRects()) textRects.push(box(r));
    }
    let parent = el.parentElement;
    while (parent && !selected.has(parent)) parent = parent.parentElement;
    const fontFamilies = cs.fontFamily.split(',').map(x => x.trim().replace(/^['"]|['"]$/g, ''));
    const unavailableFonts = fontFamilies.filter(name => !document.fonts.check(`${cs.fontSize} "${name}"`));
    const visible = cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) > 0 && rect.width > 0 && rect.height > 0;
    const outside = rect.left < 0 || rect.top < 0 || rect.right > pageWidth || rect.bottom > pageHeight;
    return {
      element_id: idOf(el), tag: el.tagName.toLowerCase(),
      element_type: el.getAttribute('data-slidex-type'),
      semantic_role: el.getAttribute('data-slidex-role') || el.getAttribute('role') || 'unknown',
      text: (el.innerText || el.textContent || '').trim(), parent_id: parent ? idOf(parent) : null,
      bbox: box(rect), visible_bbox: intersection(rect), text_bboxes: textRects,
      client_width: el.clientWidth, client_height: el.clientHeight,
      scroll_width: el.scrollWidth, scroll_height: el.scrollHeight,
      visible, partially_outside_page: outside,
      clipped: ['hidden', 'clip'].includes(cs.overflow) && (el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight),
      stacking_order: Number.parseInt(cs.zIndex, 10) || index,
      computed_style: Object.fromEntries(['fontFamily','fontSize','fontWeight','fontStyle','color','backgroundColor','borderColor','overflow','overflowX','overflowY','display','visibility','opacity','zIndex','transform','position','objectFit','clipPath','boxShadow'].map(k => [k, cs[k]])),
      font_fallback: unavailableFonts,
      image: el.tagName === 'IMG' ? {src: el.currentSrc || el.src, natural_width: el.naturalWidth,
        natural_height: el.naturalHeight, complete: el.complete, loaded: el.complete && el.naturalWidth > 0,
        object_fit: cs.objectFit, clip_path: cs.clipPath} : {}
    };
  });
}
"""


@dataclass(frozen=True)
class BrowserObservation:
    """Outputs produced from one deterministic page load."""

    computed_ir: ComputedSlideIR
    screenshot_path: Path
    pdf_path: Path
    overlay_path: Path | None = None


def deterministic_fallback_id(dom_path: str) -> str:
    """Generate a stable fallback ID from an element's structural DOM path."""
    digest = hashlib.sha256(dom_path.encode()).hexdigest()[:16]
    return f"auto-{digest}"


def validate_element_ids(ids: Iterable[str]) -> list[str]:
    """Validate explicit IDs and return warnings for missing IDs."""
    seen: set[str] = set()
    warnings: list[str] = []
    for index, element_id in enumerate(ids):
        if not element_id or element_id.isspace():
            warnings.append(f"element {index} is missing data-slidex-id")
            continue
        if element_id in seen:
            raise ValueError(f"duplicate data-slidex-id: {element_id}")
        seen.add(element_id)
    return warnings


def extract_declared_ir(
    html: str | Path,
    *,
    slide_id: str | None = None,
    page_width: float = 1280,
    page_height: float = 720,
    global_css: str | Path | None = None,
) -> DeclaredSlideIR:
    """Extract pipeline-owned structure without inferring semantics from pixels."""
    source_path = html if isinstance(html, Path) else None
    markup = html.read_text(encoding="utf-8") if source_path else str(html)
    soup = BeautifulSoup(markup, "lxml")
    css = "\n".join(tag.get_text() for tag in soup.find_all("style"))
    if global_css is not None:
        css_path = Path(global_css)
        css += "\n" + (
            css_path.read_text(encoding="utf-8")
            if css_path.exists()
            else str(global_css)
        )

    candidates = [node for node in soup.select(_INSPECTABLE) if isinstance(node, Tag)]
    explicit_ids = [str(node.get("data-slidex-id", "")) for node in candidates]
    warnings = validate_element_ids(explicit_ids)
    id_by_node: dict[int, str] = {}
    for node in candidates:
        explicit = str(node.get("data-slidex-id", "")).strip()
        id_by_node[id(node)] = explicit or deterministic_fallback_id(_dom_path(node))

    roots: list[SlideElement] = []
    element_by_node: dict[int, SlideElement] = {}
    for node in candidates:
        parent = node.parent
        while isinstance(parent, Tag) and id(parent) not in id_by_node:
            parent = parent.parent
        parent_id = id_by_node.get(id(parent)) if isinstance(parent, Tag) else None
        style = _parse_inline_style(str(node.get("style", "")))
        bbox = _declared_bbox(style, page_width, page_height)
        element = SlideElement(
            element_id=id_by_node[id(node)],
            tag=node.name,
            element_type=node.get("data-slidex-type"),
            semantic_role=node.get("data-slidex-role") or node.get("role") or "unknown",
            text=node.get_text(" ", strip=True),
            bbox=bbox,
            style={
                **style,
                "classes": list(node.get("class", [])),
                "assets": _asset_refs(node),
            },
            parent_id=parent_id,
        )
        element_by_node[id(node)] = element
        if parent_id is None:
            roots.append(element)
        else:
            element_by_node[id(parent)].children.append(element)

    if soup.find("script"):
        warnings.append("dynamic scripts make the source potentially non-reproducible")
    remote_refs = [
        ref
        for node in soup.find_all(True)
        for ref in _asset_refs(node)
        if _REMOTE_URL.search(ref)
    ]
    if remote_refs:
        warnings.append(
            f"remote resources are not reproducible: {', '.join(sorted(set(remote_refs)))}"
        )
    if _REMOTE_URL.search(css) or "@import" in css:
        warnings.append("remote CSS or font dependencies are not reproducible")

    body = soup.body
    resolved_slide_id = slide_id or (
        str(body.get("data-slide-id")) if body and body.get("data-slide-id") else None
    )
    if not resolved_slide_id:
        resolved_slide_id = source_path.stem if source_path else "slide-1"
    return DeclaredSlideIR(
        slide_id=resolved_slide_id,
        page_width=page_width,
        page_height=page_height,
        elements=roots,
        containers=[
            element.element_id
            for element in element_by_node.values()
            if element.tag in {"section", "article", "table"}
        ],
        theme_tokens=_theme_tokens(css),
        expected_roles={
            element.element_id: element.semantic_role or "unknown"
            for element in element_by_node.values()
        },
        warnings=warnings,
    )


class BrowserPool:
    """Reuse one Chromium process while bounding simultaneously open pages."""

    def __init__(self, *, max_pages: int = 4) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        self._semaphore = asyncio.Semaphore(max_pages)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def start(self) -> BrowserPool:
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True, args=["--disable-gpu", "--no-sandbox"]
            )
        return self

    async def context(self, *, width: int, height: int) -> BrowserContext:
        await self.start()
        await self._semaphore.acquire()
        assert self._browser is not None
        try:
            context = await self._browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                locale="en-US",
                timezone_id="UTC",
                color_scheme="light",
                reduced_motion="reduce",
            )
        except BaseException:
            self._semaphore.release()
            raise
        return context

    async def release(self, context: BrowserContext) -> None:
        try:
            await context.close()
        finally:
            self._semaphore.release()

    @property
    def version(self) -> str:
        if self._browser is None:
            raise RuntimeError("browser pool is not started")
        return self._browser.version

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self) -> BrowserPool:
        return await self.start()

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class BrowserObserver:
    """Create computed IR, PNG, PDF, and an optional overlay from one page load."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        pool: BrowserPool | None = None,
        cache: ContentCache | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.pool = pool
        self.cache = cache

    async def observe(
        self,
        html_path: str | Path,
        output_dir: str | Path,
        *,
        slide_id: str | None = None,
        debug_overlay: bool = False,
    ) -> BrowserObservation:
        source = Path(html_path).resolve()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cache_key = ContentCache.key(
            "browser", source.read_bytes().hex(), self.width, self.height, slide_id
        )
        if self.cache is not None:
            cached_ir = self.cache.get_json("browser-ir", cache_key)
            cached_png = self.cache.get_bytes("browser-render", cache_key, ".png")
            cached_pdf = self.cache.get_bytes("browser-render", cache_key, ".pdf")
            if (
                cached_ir is not None
                and cached_png is not None
                and cached_pdf is not None
            ):
                screenshot = output / "render.png"
                pdf = output / "render.pdf"
                screenshot.write_bytes(cached_png)
                pdf.write_bytes(cached_pdf)
                return BrowserObservation(
                    ComputedSlideIR.model_validate(cached_ir), screenshot, pdf, None
                )
        console_errors: list[str] = []
        page_errors: list[str] = []
        resource_errors: list[str] = []
        owned_pool = self.pool is None
        pool = self.pool or BrowserPool(max_pages=1)
        context = await pool.context(width=self.width, height=self.height)
        try:
            page = await context.new_page()
            page.on(
                "console",
                lambda message: (
                    console_errors.append(message.text)
                    if message.type == "error"
                    else None
                ),
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "requestfailed",
                lambda request: resource_errors.append(
                    f"{request.url}: {request.failure}"
                ),
            )
            await page.goto(
                Path(html_path).resolve().as_uri(), wait_until="networkidle"
            )
            await page.evaluate(
                "document.fonts ? document.fonts.ready : Promise.resolve()"
            )
            await page.wait_for_function(
                "[...document.images].every(img => img.complete)"
            )
            paths = await page.evaluate(
                """selector => [...document.querySelectorAll(selector)].map(el => {
                const parts=[]; while(el && el.nodeType===Node.ELEMENT_NODE){
                const siblings=[...el.parentElement?.children||[]].filter(x=>x.tagName===el.tagName);
                parts.unshift(`${el.tagName.toLowerCase()}[${Math.max(1,siblings.indexOf(el)+1)}]`);
                el=el.parentElement;} return parts.join('/'); })""",
                _INSPECTABLE,
            )
            fallback_ids = [deterministic_fallback_id(path) for path in paths]
            await page.evaluate(
                """([selector, ids]) => [...document.querySelectorAll(selector)].forEach(
                (el, index) => { if (!el.dataset.slidexId?.trim()) el.dataset.slidexId = ids[index]; })""",
                [_INSPECTABLE, fallback_ids],
            )
            raw = await page.evaluate(_OBSERVE_SCRIPT, _INSPECTABLE)
            validate_element_ids([item["element_id"] for item in raw])
            screenshot = output / "render.png"
            pdf = output / "render.pdf"
            await page.screenshot(path=screenshot, full_page=False)
            await page.pdf(
                path=pdf,
                width=f"{self.width}px",
                height=f"{self.height}px",
                print_background=True,
            )
            overlay = output / "overlay.png" if debug_overlay else None
            if overlay:
                await _draw_overlay(page, raw)
                await page.screenshot(path=overlay, full_page=False)
            ir = _build_computed_ir(
                raw,
                slide_id or Path(html_path).stem,
                self.width,
                self.height,
                pool.version,
                console_errors,
                page_errors,
                resource_errors,
            )
            if self.cache is not None and overlay is None:
                self.cache.put_json("browser-ir", cache_key, ir)
                self.cache.put_bytes(
                    "browser-render", cache_key, ".png", screenshot.read_bytes()
                )
                self.cache.put_bytes(
                    "browser-render", cache_key, ".pdf", pdf.read_bytes()
                )
            return BrowserObservation(ir, screenshot, pdf, overlay)
        finally:
            await pool.release(context)
            if owned_pool:
                await pool.close()


def _build_computed_ir(
    raw: list[dict[str, Any]],
    slide_id: str,
    width: int,
    height: int,
    browser_version: str,
    console_errors: list[str],
    page_errors: list[str],
    resource_errors: list[str],
) -> ComputedSlideIR:
    by_id: dict[str, ComputedSlideElement] = {}
    roots: list[ComputedSlideElement] = []
    warnings: list[str] = []
    for item in raw:
        if item["element_id"].startswith("auto-"):
            warnings.append(
                f"generated fallback ID for {item['tag']}: {item['element_id']}"
            )
        item["bbox"] = ObservedBoundingBox.model_validate(item["bbox"])
        item["visible_bbox"] = (
            ObservedBoundingBox.model_validate(item["visible_bbox"])
            if item["visible_bbox"]
            else None
        )
        item["text_bboxes"] = [
            ObservedBoundingBox.model_validate(box) for box in item["text_bboxes"]
        ]
        element = ComputedSlideElement.model_validate({**item, "children": []})
        by_id[element.element_id] = element
        if element.parent_id and element.parent_id in by_id:
            by_id[element.parent_id].children.append(element)
        else:
            roots.append(element)
    ready = (
        not page_errors
        and not resource_errors
        and all(item.get("image", {}).get("loaded", True) for item in raw)
    )
    return ComputedSlideIR(
        slide_id=slide_id,
        page_width=width,
        page_height=height,
        elements=roots,
        browser="Chromium",
        browser_version=browser_version,
        warnings=warnings,
        console_errors=console_errors,
        page_errors=page_errors,
        resource_errors=resource_errors,
        render_ready=ready,
    )


async def _draw_overlay(page: Page, raw: list[dict[str, Any]]) -> None:
    await page.evaluate(
        """items => { const root=document.createElement('div'); root.id='slidex-debug-overlay';
        Object.assign(root.style,{position:'fixed',inset:'0',zIndex:'2147483647',pointerEvents:'none'});
        for(const item of items){const b=item.bbox,d=document.createElement('div');
        Object.assign(d.style,{position:'absolute',left:b.x+'px',top:b.y+'px',width:b.width+'px',height:b.height+'px',border:'1px solid #ff00ff',boxSizing:'border-box',color:'#ff00ff',font:'10px sans-serif'});
        d.textContent=item.element_id;root.appendChild(d);} document.body.appendChild(root); }""",
        raw,
    )


def _dom_path(node: Tag) -> str:
    parts: list[str] = []
    current: Tag | None = node
    while current is not None and current.name != "[document]":
        siblings = [
            item
            for item in (current.parent.children if current.parent else [])
            if isinstance(item, Tag) and item.name == current.name
        ]
        parts.append(f"{current.name}[{siblings.index(current) + 1}]")
        current = current.parent if isinstance(current.parent, Tag) else None
    return "/".join(reversed(parts))


def _parse_inline_style(style: str) -> dict[str, str]:
    return {
        key.strip().lower(): value.strip()
        for declaration in style.split(";")
        if ":" in declaration
        for key, value in [declaration.split(":", 1)]
    }


def _declared_bbox(
    style: dict[str, str], width: float, height: float
) -> BoundingBox | None:
    values = [_pixels(style.get(key)) for key in ("left", "top", "width", "height")]
    if any(value is None for value in values):
        return None
    x, y, box_width, box_height = values
    if (
        x < 0
        or y < 0
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width
        or y + box_height > height
    ):
        return None
    return BoundingBox(
        x=x,
        y=y,
        width=box_width,
        height=box_height,
        page_width=width,
        page_height=height,
    )


def _pixels(value: str | None) -> float | None:
    if value is None or not value.rstrip().endswith("px"):
        return None
    try:
        return float(value.rstrip()[:-2])
    except ValueError:
        return None


def _asset_refs(node: Tag) -> list[str]:
    return [str(node[attr]) for attr in ("src", "href", "poster") if node.get(attr)]


def _theme_tokens(css: str) -> dict[str, Any]:
    return {
        "variables": dict(_CSS_VAR.findall(css)),
        "palette": sorted(set(_CSS_COLOR.findall(css))),
        "font_scale": sorted(set(_CSS_FONT_SIZE.findall(css))),
        "safe_area": dict(
            re.findall(r"--(?:safe|margin)-([\w-]+)\s*:\s*([^;}]+)", css)
        ),
        "grid_hints": sorted(
            set(re.findall(r"grid-template-(?:columns|rows)\s*:\s*([^;}]+)", css))
        ),
    }
