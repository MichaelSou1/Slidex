"""E2E task corpus acquisition, normalization, deduplication, and briefing.

Builds the 120-task corpus (20 pilot + 100 sealed, stratified 25/25/25/25
across academic/business/product/teaching) required by Phase 13 section
13.6. Every source is a real, redistributable, openly licensed API:

- academic: arXiv abstracts (metadata + abstract text; arxiv.org API)
- business: World Bank Open Data indicators (CC BY 4.0)
- product: Wikimedia Commons category descriptions (CC BY-SA)
- teaching: OpenStax book summaries (CC BY 4.0)

This module only prepares small, redistributable metadata/text; it never
downloads copyrighted full-text PDFs into the repository or cache in bulk.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .io import content_hash, write_immutable
from .models import RequiredFact, SourceRecord, TaskBrief

_USER_AGENT = "Slidex-Phase13-Eval/1.0 (research; contact: slidex-eval@example.invalid)"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https hosts only)
        return response.read()


@dataclass(frozen=True)
class RawTaskSource:
    """One fetched, licensed source document before brief construction."""

    task_type: str
    source_id: str
    url: str
    license: str
    title: str
    text: str
    revision: str


def _minhash_signature(text: str, num_hashes: int = 32) -> tuple[int, ...]:
    """Cheap MinHash over word shingles for near-duplicate task detection."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    shingles = {" ".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))}
    if not shingles:
        shingles = {text.lower()}
    signature = []
    for seed in range(num_hashes):
        best = min(
            int(hashlib.sha256(f"{seed}:{shingle}".encode()).hexdigest(), 16)
            for shingle in shingles
        )
        signature.append(best)
    return tuple(signature)


def _jaccard_estimate(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return matches / len(left)


def fetch_academic_sources(count: int, *, query: str = "cat:cs.CL", offset: int = 0) -> list[RawTaskSource]:
    """Fetch recent arXiv abstracts; metadata/abstracts are redistributable."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": offset,
            "max_results": count,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    raw = _http_get(f"https://export.arxiv.org/api/query?{params}")
    root = ElementTree.fromstring(raw)
    sources = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        arxiv_id = (entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or "").rsplit("/", 1)[-1]
        title = " ".join((entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").split())
        published = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS) or ""
        if not arxiv_id or not summary:
            continue
        sources.append(
            RawTaskSource(
                task_type="academic",
                source_id=f"arxiv-{arxiv_id}",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                license="arXiv.org perpetual non-exclusive license (abstract/metadata redistribution)",
                title=title,
                text=summary,
                revision=published[:10] or arxiv_id,
            )
        )
    return sources


_WORLDBANK_INDICATORS: dict[str, tuple[str, str]] = {
    "NY.GDP.MKTP.KD.ZG": (
        "GDP growth",
        "recorded an annual GDP growth rate of {value:.2f}% in {date}, based on "
        "World Bank national accounts data and OECD National Accounts data files. "
        "This indicator summarizes the year-over-year change in the volume of "
        "gross domestic product at market prices, based on constant local currency.",
    ),
    "FP.CPI.TOTL.ZG": (
        "inflation, consumer prices",
        "recorded consumer price inflation of {value:.2f}% in {date}, reflecting "
        "the annual percentage change in the cost to the average consumer of "
        "acquiring a basket of goods and services, based on World Bank data.",
    ),
    "SL.UEM.TOTL.ZS": (
        "unemployment rate",
        "reported an unemployment rate of {value:.2f}% of the total labor force "
        "in {date}, per World Bank modeled ILO estimates of the share of the "
        "labor force without work but available for and seeking employment.",
    ),
    "NE.TRD.GNFS.ZS": (
        "trade openness",
        "recorded trade (exports plus imports of goods and services) equal to "
        "{value:.2f}% of GDP in {date}, a World Bank indicator of the economy's "
        "openness to international trade.",
    ),
    "BX.KLT.DINV.WD.GD.ZS": (
        "foreign direct investment",
        "received net foreign direct investment inflows equal to {value:.2f}% "
        "of GDP in {date}, according to World Bank balance-of-payments data.",
    ),
    "NY.GNP.PCAP.CD": (
        "GNI per capita",
        "recorded a gross national income per capita of ${value:,.0f} (Atlas "
        "method, current US$) in {date}, a World Bank measure of average "
        "resident income used to classify countries by income group.",
    ),
    "SP.POP.GROW": (
        "population growth",
        "recorded an annual population growth rate of {value:.2f}% in {date}, "
        "based on World Bank demographic estimates derived from national "
        "census and vital registration data.",
    ),
    "GC.DOD.TOTL.GD.ZS": (
        "central government debt",
        "carried central government debt equal to {value:.2f}% of GDP in "
        "{date}, according to World Bank public-sector debt statistics.",
    ),
    "IT.NET.USER.ZS": (
        "internet adoption",
        "reported that {value:.2f}% of individuals used the internet in "
        "{date}, per World Bank information and communication technology "
        "indicators sourced from the International Telecommunication Union.",
    ),
    "EG.ELC.ACCS.ZS": (
        "electricity access",
        "reported that {value:.2f}% of the population had access to "
        "electricity in {date}, according to World Bank sustainable energy "
        "for all database estimates.",
    ),
}


def fetch_business_sources(count: int) -> list[RawTaskSource]:
    """Fetch World Bank Open Data narratives across several indicators (CC BY 4.0).

    Uses multiple economic indicators (not just GDP growth) so per-country
    narratives are textually distinct enough to survive near-duplicate
    filtering; a single indicator repeated across countries collapses into
    near-identical template text under MinHash deduplication.
    """
    sources: list[RawTaskSource] = []
    # Over-fetch generously: near-duplicate filtering later in the pipeline
    # discards a large fraction of same-template country narratives.
    per_indicator = max(15, ((count * 3) // max(1, len(_WORLDBANK_INDICATORS))) + 5)
    for indicator_code, (label, template) in _WORLDBANK_INDICATORS.items():
        params = urllib.parse.urlencode(
            {"date": "2021:2022", "format": "json", "per_page": max(50, per_indicator * 3)}
        )
        raw = None
        for attempt in range(3):
            try:
                raw = _http_get(
                    f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?{params}",
                    timeout=30.0,
                )
                break
            except Exception:
                if attempt == 2:
                    raw = None
                else:
                    time.sleep(2.0 * (attempt + 1))
        if raw is None:
            continue
        payload = json.loads(raw)
        records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        added = 0
        for record in records:
            if record.get("value") is None or not record.get("country", {}).get("value"):
                continue
            country = record["country"]["value"]
            value = record["value"]
            text = f"{country} {template.format(value=value, date=record['date'])}"
            sources.append(
                RawTaskSource(
                    task_type="business",
                    source_id=f"worldbank-{indicator_code}-{record['countryiso3code']}-{record['date']}",
                    url=f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}",
                    license="CC BY 4.0",
                    title=f"{country} {label}, {record['date']}",
                    text=text,
                    revision=str(payload[0].get("lastupdated", record["date"])),
                )
            )
            added += 1
            if added >= per_indicator:
                break
        if len(sources) >= count:
            break
    return sources


def fetch_product_sources(count: int, *, query: str = "technology product") -> list[RawTaskSource]:
    """Fetch Wikimedia Commons category descriptions (CC BY-SA, redistributable)."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": 14,  # Category namespace
            "srlimit": max(20, count * 2),
        }
    )
    raw = _http_get(f"https://commons.wikimedia.org/w/api.php?{params}")
    payload = json.loads(raw)
    hits = payload.get("query", {}).get("search", [])
    sources = []
    for hit in hits:
        title = hit["title"]
        snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
        if not snippet:
            continue
        sources.append(
            RawTaskSource(
                task_type="product",
                source_id=f"commons-{hit['pageid']}",
                url=f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
                license="CC BY-SA 4.0",
                title=title.removeprefix("Category:"),
                text=f"{title.removeprefix('Category:')}: {snippet}",
                revision=hit.get("timestamp", "")[:10],
            )
        )
        if len(sources) >= count:
            break
    return sources


def fetch_teaching_sources(count: int) -> list[RawTaskSource]:
    """Fetch OpenStax book summaries (CC BY 4.0 open educational resources)."""
    params = urllib.parse.urlencode(
        {"type": "books.Book", "fields": "title,description", "limit": max(20, count * 2)}
    )
    raw = _http_get(f"https://openstax.org/apps/cms/api/v2/pages/?{params}")
    payload = json.loads(raw)
    items = payload.get("items", [])
    sources = []
    for item in items:
        description = (item.get("description") or "").strip()
        title = item.get("title", "")
        if not description or not title:
            continue
        slug = item.get("meta", {}).get("slug", str(item.get("id")))
        sources.append(
            RawTaskSource(
                task_type="teaching",
                source_id=f"openstax-{slug}",
                url=item.get("meta", {}).get("html_url", "https://openstax.org"),
                license="CC BY 4.0",
                title=title,
                text=f"{title}: {description}",
                revision=str(item.get("meta", {}).get("first_published_at", ""))[:10],
            )
        )
        if len(sources) >= count:
            break
    return sources


_FETCHERS = {
    "academic": fetch_academic_sources,
    "business": fetch_business_sources,
    "product": fetch_product_sources,
    "teaching": fetch_teaching_sources,
}


def deduplicate_sources(
    sources: list[RawTaskSource], *, threshold: float = 0.6
) -> tuple[list[RawTaskSource], list[dict[str, str]]]:
    """Drop near-duplicate sources by title/text MinHash, keeping the first."""
    kept: list[RawTaskSource] = []
    kept_signatures: list[tuple[int, ...]] = []
    rejected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for source in sources:
        if source.source_id in seen_ids:
            rejected.append({"source_id": source.source_id, "reason": "duplicate source_id"})
            continue
        signature = _minhash_signature(f"{source.title} {source.text}")
        is_duplicate = any(
            _jaccard_estimate(signature, other) >= threshold for other in kept_signatures
        )
        if is_duplicate:
            rejected.append({"source_id": source.source_id, "reason": "near-duplicate text"})
            continue
        kept.append(source)
        kept_signatures.append(signature)
        seen_ids.add(source.source_id)
    return kept, rejected


def normalize_source(source: RawTaskSource, output_dir: Path) -> tuple[SourceRecord, Path]:
    """Persist a normalized Markdown copy with a page/paragraph locator map."""
    directory = output_dir / "e2e_sources" / source.task_type
    directory.mkdir(parents=True, exist_ok=True)
    markdown = f"# {source.title}\n\nSource: {source.url}\n\n{source.text}\n"
    path = directory / f"{source.source_id}.md"
    path.write_text(markdown, encoding="utf-8")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    record = SourceRecord(
        source_id=source.source_id,
        url=source.url,
        license=source.license,
        revision=source.revision or "unpinned",
        sha256=sha256,
        acquired_at=datetime.now(UTC),
        local_path=path.relative_to(output_dir).as_posix(),
        redistributable=True,
        dataset_id=f"slidex-e2e-{source.task_type}",
    )
    return record, path


_TASK_TYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "academic": {
        "audience": "conference or lab audience of domain researchers",
        "purpose": "present the paper's motivation, method, and key findings",
        "page_count": (6, 10),
        "required_sections": ["Motivation", "Method", "Results", "Conclusion"],
        "style_constraints": ["formal academic tone", "cite the source paper by title"],
    },
    "business": {
        "audience": "executives and analysts reviewing macroeconomic indicators",
        "purpose": "summarize the indicator trend and its business implication",
        "page_count": (4, 7),
        "required_sections": ["Overview", "Key Metric", "Implication"],
        "style_constraints": ["concise business tone", "cite the World Bank indicator by name"],
    },
    "product": {
        "audience": "prospective customers and partners at a product briefing",
        "purpose": "introduce the product/technology category and its relevance",
        "page_count": (5, 8),
        "required_sections": ["Overview", "Key Capabilities", "Use Cases"],
        "style_constraints": ["marketing-adjacent but factual tone"],
    },
    "teaching": {
        "audience": "undergraduate students in an introductory course",
        "purpose": "explain the textbook topic for a lecture handout",
        "page_count": (6, 9),
        "required_sections": ["Learning Objectives", "Core Concepts", "Summary"],
        "style_constraints": ["clear pedagogical tone", "define technical terms on first use"],
    },
}


def build_task_brief(source: RawTaskSource) -> TaskBrief:
    """Derive a structured brief from one normalized source's own text.

    Required facts are the source's declarative sentences; a human expert
    still verifies each brief is achievable from the source material before
    it is admitted into the frozen corpus (13.6 checklist item).
    """
    defaults = _TASK_TYPE_DEFAULTS[source.task_type]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source.text) if len(s.strip()) > 20]
    facts = [
        RequiredFact(
            fact_id=f"{source.source_id}-fact-{index + 1}",
            text=sentence,
            source_locator=f"{source.source_id}#sentence-{index + 1}",
        )
        for index, sentence in enumerate(sentences[:5])
    ]
    return TaskBrief(
        audience=defaults["audience"],
        purpose=f"{defaults['purpose']}: {source.title}",
        language="en",
        page_count=defaults["page_count"],
        required_sections=list(defaults["required_sections"]),
        required_facts=facts,
        required_visuals=[],
        style_constraints=list(defaults["style_constraints"]),
        forbidden_claims=["do not invent statistics not present in the source material"],
        acceptable_summarization="Paraphrasing is acceptable; fabricating new facts is not.",
        automatic_checks=["required_sections_present", "required_facts_retained"],
        human_rating_prompt=(
            f"Does this deck faithfully and clearly present {source.title} for the stated audience?"
        ),
    )


def build_task_corpus(
    output_dir: Path,
    *,
    per_type_sealed: int = 25,
    per_type_pilot: int = 5,
    seed: int = 13,
) -> dict[str, Any]:
    """Fetch, normalize, dedupe, and brief the full 120-task E2E corpus.

    Returns the frozen corpus manifest (also written to disk) with sealed and
    pilot splits stratified evenly across the four task types.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_type_total = per_type_sealed + per_type_pilot
    all_sources: list[RawTaskSource] = []
    fetch_errors: list[dict[str, str]] = []
    for task_type, fetcher in _FETCHERS.items():
        try:
            # Over-fetch generously: MinHash near-duplicate filtering (13.6)
            # discards a substantial fraction of same-template source text.
            fetched = fetcher(per_type_total * 3)
        except Exception as exc:  # network/API failures must be visible, not silently skipped
            fetch_errors.append({"task_type": task_type, "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_sources.extend(fetched)

    kept, rejected_duplicates = deduplicate_sources(all_sources)
    by_type: dict[str, list[RawTaskSource]] = {}
    for source in kept:
        by_type.setdefault(source.task_type, []).append(source)

    tasks: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    for task_type in _FETCHERS:
        candidates = by_type.get(task_type, [])
        if len(candidates) < per_type_total:
            shortfalls[task_type] = per_type_total - len(candidates)
        selected = candidates[:per_type_total]
        for index, source in enumerate(selected):
            record, path = normalize_source(source, output_dir)
            brief = build_task_brief(source)
            split = "pilot" if index < per_type_pilot else "sealed_test"
            task_id = hashlib.sha256(f"e2e-task:{source.source_id}".encode()).hexdigest()[:24]
            tasks.append(
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "split": split,
                    "source": record.model_dump(mode="json"),
                    "normalized_path": path.relative_to(output_dir).as_posix(),
                    "brief": brief.model_dump(mode="json"),
                }
            )

    manifest = {
        "schema_version": "1.0",
        "corpus_id": "slidex-phase13-e2e-tasks",
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
        "per_type_sealed": per_type_sealed,
        "per_type_pilot": per_type_pilot,
        "tasks": sorted(tasks, key=lambda item: item["task_id"]),
        "rejected_duplicates": rejected_duplicates,
        "fetch_errors": fetch_errors,
        "shortfalls": shortfalls,
    }
    manifest["corpus_hash"] = content_hash(manifest)
    manifest_path = output_dir / "e2e_task_corpus.json"
    if manifest_path.exists():
        write_immutable(manifest_path, manifest)
    else:
        write_immutable(manifest_path, manifest)
    return manifest
