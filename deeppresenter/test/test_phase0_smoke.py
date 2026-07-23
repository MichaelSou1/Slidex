from pathlib import Path

import pytest

from deeppresenter.utils.webview import PlaywrightConverter, convert_html_to_pptx


HTML = """<!doctype html><html><body style='margin:0;width:1280px;height:720px'><h1>Slidex smoke test</h1></body></html>"""


@pytest.mark.export
@pytest.mark.asyncio
async def test_html_to_pptx_strict_validation(tmp_path: Path) -> None:
    html = tmp_path / "slide.html"
    html.write_text(HTML, encoding="utf-8")
    output = tmp_path / "slide.pptx"
    await convert_html_to_pptx(html, output, soft_parsing=False)
    assert output.stat().st_size > 0


@pytest.mark.browser
@pytest.mark.asyncio
async def test_single_page_html_render(tmp_path: Path) -> None:
    html = tmp_path / "slide.html"
    html.write_text(HTML, encoding="utf-8")
    output = tmp_path / "slide.png"
    async with PlaywrightConverter() as converter:
        page = await converter.context.new_page()
        try:
            await page.goto(html.resolve().as_uri())
            await page.screenshot(path=output)
        finally:
            await page.close()
    assert output.stat().st_size > 0
