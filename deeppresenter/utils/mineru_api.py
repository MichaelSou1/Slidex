import asyncio
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import aiohttp


async def parse_pdf_offline(pdf_path: str, output_path: str, url: str) -> None:
    """Parse PDF using a local/compatible MinerU endpoint."""
    os.makedirs(output_path, exist_ok=True)
    pdf_path_obj = Path(pdf_path)

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field(
            "pdf",
            pdf_path_obj.read_bytes(),
            filename=pdf_path_obj.name,
            content_type="application/pdf",
        )

        async with session.post(url, data=form) as resp:
            if resp.status != 200:
                await _raise_parsedoc_error(resp)
            content = await resp.read()

    _extract_zip_bytes(content, output_path)


async def parse_pdf_online(
    pdf_path: str, output_path: str, token: str, model_version: str = "vlm"
) -> None:
    """Parse PDF using MinerU external API

    Args:
        pdf_path: PDF file path
        output_path: Output directory
        token: API Token
        model_version: Model version (vlm/pipeline)
    """
    os.makedirs(output_path, exist_ok=True)
    pdf_path = Path(pdf_path)

    async with aiohttp.ClientSession() as session:
        batch_id, upload_url, upload_headers = await _request_upload_url(
            session, pdf_path.name, pdf_path.stem[:128], model_version, token
        )

        await _upload_file(session, upload_url, pdf_path, upload_headers)

        zip_url = await _poll_result(session, batch_id, token)

        await _download_and_extract(session, zip_url, output_path)


async def _request_upload_url(
    session: aiohttp.ClientSession,
    filename: str,
    data_id: str,
    model_version: str,
    token: str,
) -> tuple[str, str, dict[str, str] | None]:
    """Request upload URL, returns (batch_id, upload_url, upload_headers)"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "files": [{"name": filename, "data_id": data_id}],
        "model_version": model_version,
    }

    async with session.post(
        "https://mineru.net/api/v4/file-urls/batch", headers=headers, json=payload
    ) as resp:
        resp.raise_for_status()
        result = await resp.json()
        if result["code"] != 0:
            raise RuntimeError(
                f"Failed to request upload URL: {result.get('msg', 'Unknown error')}"
            )

        data = result["data"]
        upload_headers = data.get("headers", [None])[0] if "headers" in data else None
        return data["batch_id"], data["file_urls"][0], upload_headers


async def _upload_file(
    session: aiohttp.ClientSession,
    upload_url: str,
    pdf_path: Path,
    headers: dict[str, str] | None = None,
) -> None:
    """Upload PDF file to OSS"""
    file_data = pdf_path.read_bytes()

    upload_headers = headers if headers else {}

    async with session.put(
        upload_url,
        data=file_data,
        headers=upload_headers,
        skip_auto_headers={"Content-Type"},
    ) as resp:
        resp.raise_for_status()


async def _poll_result(
    session: aiohttp.ClientSession, batch_id: str, token: str
) -> str:
    """Poll parsing result, returns download URL"""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    while True:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            result = await resp.json()

            if result["code"] != 0:
                raise RuntimeError(
                    f"Query failed: {result.get('msg', 'Unknown error')}"
                )

            extract = result["data"]["extract_result"][0]
            state = extract["state"]

            if state == "done":
                return extract["full_zip_url"]
            elif state == "failed":
                raise RuntimeError(
                    f"Parsing failed: {extract.get('err_msg', 'Unknown error')}"
                )

            await asyncio.sleep(5)


async def _download_and_extract(
    session: aiohttp.ClientSession, zip_url: str, output_path: str
) -> None:
    """Download and extract result"""
    async with session.get(zip_url) as resp:
        resp.raise_for_status()
        content = await resp.read()

    _extract_zip_bytes(content, output_path)


async def _raise_parsedoc_error(resp: aiohttp.ClientResponse) -> None:
    """Raise a RuntimeError with parsed error content."""
    try:
        payload = await resp.json()
    except Exception:
        payload = await resp.text()
    raise RuntimeError(payload)


def _extract_zip_bytes(
    content: bytes,
    output_path: str,
    *,
    max_files: int = 10_000,
    max_uncompressed_bytes: int = 1_000_000_000,
    max_compression_ratio: float = 100.0,
) -> None:
    """Extract a bounded archive without zip-slip, symlink, or zip-bomb entries."""
    destination = Path(output_path).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(content)
        zip_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [
                member for member in archive.infolist() if member.filename.strip()
            ]
            if len(members) > max_files:
                raise ValueError("archive contains too many entries")
            total_size = sum(member.file_size for member in members)
            if total_size > max_uncompressed_bytes:
                raise ValueError("archive exceeds uncompressed size limit")
            for member in members:
                compressed = max(member.compress_size, 1)
                if member.file_size / compressed > max_compression_ratio:
                    raise ValueError(f"suspicious compression ratio: {member.filename}")
                mode = member.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise ValueError(f"archive symlink is forbidden: {member.filename}")

            names = [member.filename for member in members]
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"archive path escapes output directory: {name}")
            top_level = {name.split("/", 1)[0] for name in names}
            prefix = (
                next(iter(top_level)) + "/"
                if len(top_level) == 1 and all("/" in name for name in names)
                else ""
            )
            for member in members:
                if member.is_dir():
                    continue
                relative = (
                    member.filename.removeprefix(prefix) if prefix else member.filename
                )
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(
                        f"archive path escapes output directory: {member.filename}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    finally:
        zip_path.unlink(missing_ok=True)
