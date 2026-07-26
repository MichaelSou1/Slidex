"""Deterministic, defect-specific PowerPoint mutations for paired evaluation data."""

from __future__ import annotations

import hashlib
import random
import zipfile
from pathlib import Path

from lxml import etree

from deeppresenter.slidex.models import DefectClass

from .models import MutationRecord

_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _seed(parent_deck_id: str, defect_class: DefectClass, variant: int) -> int:
    payload = f"slidex-mutation-v1\0{parent_deck_id}\0{defect_class.value}\0{variant}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _mutation_id(parent_deck_id: str, defect_class: DefectClass, variant: int) -> str:
    payload = f"{parent_deck_id}\0{defect_class.value}\0{variant}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _shape_id(node: etree._Element) -> str | None:
    ids = node.xpath(".//p:cNvPr/@id", namespaces=_NS)
    return str(ids[0]) if ids else None


def _shapes(root: etree._Element) -> list[etree._Element]:
    return list(root.xpath("//p:sp | //p:pic | //p:graphicFrame", namespaces=_NS))


def _xfrm(shape: etree._Element) -> etree._Element | None:
    found = shape.xpath(".//a:xfrm", namespaces=_NS)
    return found[0] if found else None


def _off_ext(shape: etree._Element) -> tuple[etree._Element, etree._Element] | None:
    transform = _xfrm(shape)
    if transform is None:
        return None
    offsets = transform.xpath("./a:off", namespaces=_NS)
    extents = transform.xpath("./a:ext", namespaces=_NS)
    return (offsets[0], extents[0]) if offsets and extents else None


def _text_nodes(root: etree._Element) -> list[etree._Element]:
    return [node for node in root.xpath("//a:t", namespaces=_NS) if (node.text or "").strip()]


def _geometry_mutation(
    root: etree._Element, defect: DefectClass, rng: random.Random
) -> tuple[str, dict[str, object], str | None]:
    candidates = [(shape, pair) for shape in _shapes(root) if (pair := _off_ext(shape))]
    if defect in {DefectClass.G1, DefectClass.G7}:
        text_candidates = [item for item in candidates if item[0].xpath(".//a:t", namespaces=_NS)]
        if text_candidates:
            candidates = text_candidates
    if not candidates:
        raise ValueError("no mutable native geometry found")
    shape, (off, ext) = candidates[rng.randrange(len(candidates))]
    target = _shape_id(shape)
    parameters: dict[str, object] = {}
    if defect in {DefectClass.G1, DefectClass.G7}:
        old = int(ext.get("cy", "0"))
        factor = 0.35 if defect is DefectClass.G1 else 0.65
        ext.set("cy", str(max(1, int(old * factor))))
        parameters = {"attribute": "cy", "before": old, "after": int(ext.get("cy"))}
        return ("shrink_text_container" if defect is DefectClass.G1 else "induce_containment_overflow", parameters, target)
    if defect is DefectClass.G2:
        if len(candidates) < 2:
            raise ValueError("G2 requires two mutable shapes")
        other, (other_off, _) = next(item for item in candidates if item[0] is not shape)
        before = {"x": int(off.get("x", "0")), "y": int(off.get("y", "0"))}
        off.set("x", other_off.get("x", "0"))
        off.set("y", other_off.get("y", "0"))
        return "create_element_overlap", {"before": before, "after": {"x": int(off.get("x")), "y": int(off.get("y"))}, "anchor_id": _shape_id(other)}, target
    if defect is DefectClass.G3:
        old = int(off.get("x", "0"))
        delta = 127000
        off.set("x", str(old + delta))
        return "offset_alignment", {"axis": "x", "before": old, "delta": delta}, target
    if defect is DefectClass.G6:
        old = int(off.get("x", "0"))
        width = int(ext.get("cx", "1"))
        off.set("x", str(-max(12700, min(width // 4, 457200))))
        return "cross_safe_margin", {"axis": "x", "before": old, "after": int(off.get("x"))}, target
    raise ValueError(f"unsupported geometry mutation: {defect.value}")


def _font_mutation(root: etree._Element, rng: random.Random) -> tuple[str, dict[str, object], str | None]:
    nodes = root.xpath("//a:rPr | //a:defRPr", namespaces=_NS)
    if not nodes:
        raise ValueError("no mutable font size found")
    node = nodes[rng.randrange(len(nodes))]
    old = int(node.get("sz", "1800"))
    new = max(800, old + 800)
    node.set("sz", str(new))
    return "change_font_scale", {"before": old, "after": new}, None


def _color_mutation(root: etree._Element, rng: random.Random) -> tuple[str, dict[str, object], str | None]:
    nodes = root.xpath("//a:srgbClr | //a:schemeClr", namespaces=_NS)
    if not nodes:
        raise ValueError("no mutable slide color found")
    node = nodes[rng.randrange(len(nodes))]
    old = node.get("val", "000000")
    new = "FF00FF" if old.upper() != "FF00FF" else "00FFFF"
    node.tag = f"{{{_NS['a']}}}srgbClr"
    node.attrib.clear()
    node.set("val", new)
    return "replace_off_palette_color", {"before": old, "after": new}, None


def _semantic_mutation(root: etree._Element, defect: DefectClass, rng: random.Random) -> tuple[str, dict[str, object], str | None]:
    texts = _text_nodes(root)
    if not texts:
        raise ValueError("no mutable slide text found")
    node = texts[rng.randrange(len(texts))]
    old = node.text or ""
    if defect is DefectClass.S1:
        new, operator = "This body discusses an unrelated and contradictory topic.", "replace_body_with_mismatch"
    elif defect is DefectClass.S3:
        token = next((part for part in old.split() if len(part) >= 4), old)
        new, operator = old.replace(token, f"{token}-variant", 1), "introduce_terminology_variant"
    elif defect is DefectClass.S4:
        new, operator = (old + " " + " ".join([old] * 8)).strip(), "increase_density"
    elif defect is DefectClass.S5:
        shape = node.xpath("ancestor::p:sp[1]", namespaces=_NS)
        if not shape:
            raise ValueError("no removable logic section found")
        target = _shape_id(shape[0])
        shape[0].getparent().remove(shape[0])
        return "remove_logic_section", {"removed_text_sha256": hashlib.sha256(old.encode()).hexdigest()}, target
    elif defect is DefectClass.S6:
        new, operator = "The pictured evidence shows the exact opposite of this claim.", "contradict_image_caption"
    else:
        raise ValueError(f"unsupported semantic mutation: {defect.value}")
    node.text = new
    shape = node.xpath("ancestor::p:sp[1]", namespaces=_NS)
    return operator, {"before_sha256": hashlib.sha256(old.encode()).hexdigest(), "after_sha256": hashlib.sha256(new.encode()).hexdigest()}, _shape_id(shape[0]) if shape else None


def _mutate_slide_xml(data: bytes, defect: DefectClass, rng: random.Random) -> tuple[bytes, str, dict[str, object], str | None]:
    root = etree.fromstring(data)
    if defect in {DefectClass.G1, DefectClass.G2, DefectClass.G3, DefectClass.G6, DefectClass.G7}:
        operator, params, target = _geometry_mutation(root, defect, rng)
    elif defect is DefectClass.G4:
        operator, params, target = _font_mutation(root, rng)
    elif defect is DefectClass.G5:
        operator, params, target = _color_mutation(root, rng)
    else:
        operator, params, target = _semantic_mutation(root, defect, rng)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True), operator, params, target


def mutate_pptx(
    source: Path,
    target: Path,
    defect_class: DefectClass,
    parent_deck_id: str,
    variant: int = 0,
) -> MutationRecord:
    """Create one deterministic single-defect candidate without modifying the source."""
    seed = _seed(parent_deck_id, defect_class, variant)
    rng = random.Random(seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    slide_names: list[str]
    with zipfile.ZipFile(source) as archive:
        if defect_class is DefectClass.S2:
            presentation = "ppt/presentation.xml"
            root = etree.fromstring(archive.read(presentation))
            slide_ids = root.xpath("//p:sldIdLst/p:sldId", namespaces=_NS)
            if len(slide_ids) < 2:
                raise ValueError("S2 requires at least two slides")
            first = rng.randrange(len(slide_ids) - 1)
            parent = slide_ids[first].getparent()
            right = slide_ids[first + 1]
            parent.remove(right)
            parent.insert(first, right)
            mutated_presentation = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, "w") as output:
                for info in archive.infolist():
                    output.writestr(info, mutated_presentation if info.filename == presentation else archive.read(info.filename))
            return MutationRecord(
                mutation_id=_mutation_id(parent_deck_id, defect_class, variant),
                defect_class=defect_class,
                operator="swap_adjacent_sections",
                parameters={"first_index": first, "second_index": first + 1},
                seed=seed,
            )
        slide_names = sorted(name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
        if not slide_names:
            raise ValueError("PPTX contains no slide XML")
        selected = slide_names[rng.randrange(len(slide_names))]
        with zipfile.ZipFile(target, "w") as output:
            result: tuple[str, dict[str, object], str | None] | None = None
            for info in archive.infolist():
                data = archive.read(info.filename)
                if info.filename == selected:
                    data, operator, parameters, element_id = _mutate_slide_xml(data, defect_class, rng)
                    result = operator, {"slide_part": selected, **parameters}, element_id
                output.writestr(info, data)
    if result is None:
        target.unlink(missing_ok=True)
        raise ValueError("mutation did not change a slide")
    operator, parameters, element_id = result
    return MutationRecord(
        mutation_id=_mutation_id(parent_deck_id, defect_class, variant),
        defect_class=defect_class,
        operator=operator,
        parameters=parameters,
        seed=seed,
        target_element_id=element_id,
    )
