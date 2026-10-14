"""Small, deterministic helpers for parsing structured bibliography records."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any
import xml.etree.ElementTree as ET


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
ARXIV_PATTERN = re.compile(
    r"(?:arxiv\s*(?:preprint)?\s*:|abs/)\s*(\d{4}\.\d{4,5})(?:v(\d+))?",
    re.IGNORECASE,
)


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def _normalized_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _clean_grobid_title(value: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value.strip())
    if len(parts) > 1 and len(parts[0]) > 100 and 8 <= len(parts[-1]) <= 200:
        return parts[-1].strip()
    return value.strip()


def _reference_year(reference_text: str, reference: ET.Element) -> int | None:
    for date in reference.findall(".//tei:date", TEI_NS):
        candidate = date.attrib.get("when") or _element_text(date)
        match = re.search(r"\b(?:19|20)\d{2}\b", candidate)
        if match:
            return int(match.group(0))
    years = re.findall(r"\b(?:19|20)\d{2}\b", reference_text)
    return int(years[-1]) if years else None


def parse_grobid_paper(tei_xml: str) -> dict[str, Any]:
    """Extract root-paper metadata and bibliography from a GROBID TEI response."""
    root = ET.fromstring(tei_xml)
    title = _clean_grobid_title(_element_text(
        root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", TEI_NS)
    ))
    authors = []
    for author in root.findall(".//tei:teiHeader/tei:fileDesc/tei:sourceDesc//tei:author", TEI_NS):
        given = _element_text(author.find("./tei:persName/tei:forename", TEI_NS))
        family = _element_text(author.find("./tei:persName/tei:surname", TEI_NS))
        name = " ".join(part for part in (given, family) if part)
        if name and name not in authors:
            authors.append(name)

    doi = None
    arxiv_id = None
    for identifier in root.findall(".//tei:teiHeader//tei:idno", TEI_NS):
        identifier_type = identifier.attrib.get("type", "").lower()
        if "doi" in identifier_type and not doi:
            doi = _element_text(identifier).removeprefix("https://doi.org/").removeprefix("doi:").strip()
        elif ("arxiv" in identifier_type or not identifier_type) and not arxiv_id:
            match = ARXIV_PATTERN.search(_element_text(identifier))
            if match:
                arxiv_id = match.group(1)

    abstract = _element_text(root.find(".//tei:teiHeader/tei:profileDesc/tei:abstract", TEI_NS))
    keywords = [
        _element_text(term)
        for term in root.findall(".//tei:teiHeader/tei:profileDesc/tei:textClass/tei:keywords//tei:term", TEI_NS)
        if _element_text(term)
    ]
    return {
        "title": title or None,
        "authors": authors,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "abstract": abstract,
        "keywords": keywords,
        "references": parse_grobid_references(tei_xml),
    }


def parse_grobid_references(tei_xml: str) -> list[dict[str, Any]]:
    """Parse GROBID TEI bibliography entries without discarding the raw citation."""
    root = ET.fromstring(tei_xml)
    bibliography = root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)
    references: list[dict[str, Any]] = []
    for index, item in enumerate(bibliography, start=1):
        analytic_title = _element_text(item.find("./tei:analytic/tei:title", TEI_NS))
        monograph_title = _element_text(item.find("./tei:monogr/tei:title", TEI_NS))
        title = analytic_title or monograph_title
        raw = _element_text(item.find('./tei:note[@type="raw_reference"]', TEI_NS))
        authors: list[str] = []
        for author in item.findall(".//tei:author", TEI_NS):
            given = _element_text(author.find("./tei:persName/tei:forename", TEI_NS))
            family = _element_text(author.find("./tei:persName/tei:surname", TEI_NS))
            name = " ".join(part for part in (given, family) if part)
            if name and name not in authors:
                authors.append(name)

        doi = None
        for identifier in item.findall(".//tei:idno", TEI_NS):
            if "doi" in identifier.attrib.get("type", "").lower():
                doi = _element_text(identifier).removeprefix("https://doi.org/").removeprefix("doi:").strip()
                break

        arxiv_match = ARXIV_PATTERN.search(f"{raw} {_element_text(item)}")
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        arxiv_version = int(arxiv_match.group(2)) if arxiv_match and arxiv_match.group(2) else None
        raw = raw or _element_text(item)
        references.append({
            "number": index,
            "grobid_id": item.attrib.get(XML_ID),
            "title": title or None,
            "authors": authors,
            "year": _reference_year(raw, item),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "arxiv_version": arxiv_version,
            "raw_reference": raw,
        })
    if not references:
        raise ValueError("GROBID returned no structured bibliography entries")
    return references


def title_similarity(reference_title: str, candidate_title: str) -> float:
    """Return a title score while allowing a cited short title to omit a subtitle."""
    expected = _normalized_title(reference_title)
    candidate = _normalized_title(candidate_title)
    if not expected or not candidate:
        return 0.0
    if expected == candidate:
        return 1.0
    expected_tokens = set(expected.split())
    candidate_tokens = set(candidate.split())
    overlap = len(expected_tokens & candidate_tokens)
    recall = overlap / max(1, len(expected_tokens))
    precision = overlap / max(1, len(candidate_tokens))
    token_score = 2 * recall * precision / max(0.0001, recall + precision)
    sequence_score = SequenceMatcher(None, expected, candidate).ratio()
    return max(token_score, sequence_score)


def first_author_matches(reference_authors: list[str], candidate_authors: list[str]) -> bool:
    if not reference_authors or not candidate_authors:
        return False
    reference_names = _normalized_title(reference_authors[0]).split()
    candidate_names = _normalized_title(candidate_authors[0]).split()
    if not reference_names or not candidate_names:
        return False
    reference_surname = reference_names[-1]
    candidate_surname = candidate_names[-1]
    return bool(reference_surname and reference_surname == candidate_surname)

