r"""Inventory every PDF under PDF/ into data/local_sources.json.

Offline, no network. Unknown bibliographic fields stay null. Filename hints
are recorded separately and never treated as verified metadata.

Usage (PowerShell, from repo root):
  $env:PYTHONPATH = "src"
  .venv/Scripts/python.exe scripts/inventory_local_pdfs.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = ROOT / "PDF"
OUT_PATH = ROOT / "data" / "local_sources.json"
MANIFEST_DIR = ROOT / "data" / "private" / "traecode_20261014"
MANIFEST_PATH = MANIFEST_DIR / "geoscience_manifest.json"

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# ---------------------------------------------------------------------------
# Content-based geoscience classification (task D).
#
# The decision is driven by the words actually found in the title/first pages,
# NOT by a fixed filename blacklist: a filename hint may raise a candidate but
# never decides eligibility on its own (AGENTS.md / task brief §5).
# ---------------------------------------------------------------------------

# Strong subject vocabulary of geology / petrology / mineralogy / geochemistry /
# tectonics / geophysics / Earth evolution. Matched case-insensitively.
_GEOSCIENCE_TERMS = (
    "geolog", "petrolog", "mineral", "geochem", "isotop", "zircon", "magma",
    "magmatic", "basalt", "granit", "tectonic", "subduction", "crust", "mantle",
    "lithospher", "seism", "orogen", "metamorph", "volcan", "plate", "uplift",
    "tibet", "gangdese", "himalaya", "geophys", "geodynamic", "igneous",
    "sedimentary", "continental", "archean", "proterozoic", "paleozoic",
    "mesozoic", "cenozoic", "eocene", "rare earth element", "trace element",
    "mid-ocean ridge", "moho", "adakite", "eclogite", "tonalite", "granulite",
    "ophiolite", "terrane", "batholith", "earthquake", "paleomagnet", "crater",
    "hydrothermal", "mineralog", "geochronolog", "earth science", "planetary",
    "deep earth", "upper mantle", "lower crust", "slab", "arc magma",
    "deformation", "sediment", "tethys", "pluton", "pyroxene", "olivine",
    # Chinese terms (substring roots), so Chinese-language papers classify by
    # their own content rather than falling through the English vocabulary.
    "地质", "地球化学", "地球物理", "岩石", "岩浆", "岩基", "地壳", "地幔",
    "壳幔", "构造", "青藏", "高原", "造山", "俯冲", "变质", "花岗", "玄武",
    "同位素", "地震", "火山", "矿物", "矿床", "年代", "成矿", "大陆动力学",
    "岩石圈", "板块", "藏南", "冈底斯", "喜马拉雅", "特提斯", "断裂", "盆地",
    "沉积", "古高度", "隆升", "碰撞", "蛇绿", "橄榄", "辉长", "斜长石",
    "锆石", "稀土", "微量元素",
)
# Machine learning / modelling applied *to* a geoscience problem (task §5).
_ML_GEO_CUES = ("machine learning", "neural network", "deep learning")
_ML_GEO_SUBJECTS = (
    "crust", "water content", "mid-ocean ridge", "melting pressure", "tectonic",
    "thickness", "mantle", "seismic", "geochem", "volcan",
)
# Non-geoscience cues: psychology, self-help, hiking/travel, generic writing or
# generic ML textbooks. A mountain name alone never makes a file geoscience.
_NON_GEO_TERMS = (
    "psycholog", "relationship", "emotion", "self-help", "well-being",
    "mindful", "meditation", "walking", "trekking", "hike", "day walks",
    "hut-to-hut", "maritime alps", "austria", "routes", "research writing",
    "academic writing", "essay writing", "grammar", "machine learning theory",
    "supervised learning textbook", "python for beginners",
)



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filename_hints(stem: str) -> dict:
    """Parse author/year-style hints from the filename only (unverified)."""
    hints: dict = {
        "title_from_filename": stem,
        "authors_from_filename": [],
        "year_from_filename": None,
    }
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1].strip()
        if tail and not tail.isdigit() and len(tail) <= 20 and not re.search(r"\d{4}", tail):
            if re.fullmatch(r"[一-鿿]{2,4}", tail) or re.fullmatch(r"[A-Za-z][A-Za-z .-]{1,30}", tail):
                hints["authors_from_filename"] = [tail]
    year_match = re.search(r"(?:^|[^\d])((?:19|20)\d{2})(?:[^\d]|$)", stem)
    if year_match:
        hints["year_from_filename"] = int(year_match.group(1))
    return hints


def first_page_biblio(pdf_path: Path, max_pages: int = 2) -> dict:
    """Read title-ish text from the first pages. Never invents values."""
    out = {
        "pdf_page_count": None,
        "pdf_title": None,
        "pdf_author": None,
        "extractable": False,
        "first_pages_chars": 0,
        "first_pages_sample": "",
        "first_pages_text": "",
        "doi_in_text": None,
        "year_in_text": None,
        "extract_note": None,
    }
    try:
        reader = PdfReader(str(pdf_path))
        out["pdf_page_count"] = len(reader.pages)
        meta = reader.metadata
        if meta:
            title = meta.get("/Title")
            author = meta.get("/Author")
            if isinstance(title, str) and title.strip() and len(title.strip()) > 2:
                out["pdf_title"] = title.strip()[:300]
            if isinstance(author, str) and author.strip() and len(author.strip()) > 2:
                out["pdf_author"] = author.strip()[:300]
        texts = []
        for page in reader.pages[:max_pages]:
            try:
                texts.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                out["extract_note"] = f"page extract error: {exc}"
        joined = "\n".join(texts).strip()
        out["first_pages_chars"] = len(joined)
        out["extractable"] = len(re.sub(r"\s+", "", joined)) >= 80
        out["first_pages_sample"] = re.sub(r"\s+", " ", joined)[:400]
        out["first_pages_text"] = joined[:6000]
        doi_match = DOI_RE.search(joined)
        if doi_match:
            out["doi_in_text"] = doi_match.group(0)
        years = [int(m.group(0)) for m in YEAR_RE.finditer(joined)]
        plausible = [y for y in years if 1950 <= y <= 2100]
        if plausible:
            out["year_in_text"] = max(set(plausible), key=plausible.count)
    except Exception as exc:  # noqa: BLE001
        out["extract_note"] = f"open/read error: {exc}"
    return out


def _scan_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Concatenate the first ``max_pages`` pages of extractable text (lowercased
    later); used for content-based classification, never stored verbatim."""
    try:
        reader = PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages[:max_pages]:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(chunks)
    except Exception:  # noqa: BLE001
        return ""


def classify_geoscience(text: str, filename: str) -> tuple[str, str, int]:
    """Return ``(decision, reason, inspected_pages_estimate)``.

    ``include`` / ``exclude`` are the confident outcomes; ``uncertain`` is used
    when the content signals are mixed so the file waits for human review and
    stays out of the new library.
    """
    lowered = (text or "").lower()
    name = filename.lower()
    # Body vs filename separated — a filename hint alone never proves a subject.
    body_geo_hits = sorted({t for t in _GEOSCIENCE_TERMS if t in lowered})
    name_geo_hits = sorted({t for t in _GEOSCIENCE_TERMS if t in name})
    body_non_hits = sorted({t for t in _NON_GEO_TERMS if t in lowered})
    name_non_hits = sorted({t for t in _NON_GEO_TERMS if t in name})
    all_geo = sorted(set(body_geo_hits) | set(name_geo_hits))
    all_non = sorted(set(body_non_hits) | set(name_non_hits))
    ml_cues = [t for t in _ML_GEO_CUES if t in lowered]
    ml_subjects = [t for t in _ML_GEO_SUBJECTS if t in lowered]

    reason_bits = []
    if body_geo_hits:
        reason_bits.append("正文地学术语：" + "、".join(body_geo_hits[:8]))
    if name_geo_hits:
        reason_bits.append("文件名地学术语：" + "、".join(name_geo_hits[:4]))
    if ml_cues and ml_subjects:
        reason_bits.append("地学应用机器学习：" + "、".join(ml_cues + ml_subjects[:4]))
    if all_non:
        reason_bits.append("非地学线索：" + "、".join(all_non[:8]))
    reason = "；".join(reason_bits) or "无明确地学或非地学术语"

    # Visible text volume (spaces removed) decides the ambiguous tail case; a
    # file with plenty of prose but no subject vocabulary at all is a textbook /
    # non-literal-topic document, not geoscience literature.
    text_len = len(re.sub(r"\s+", "", lowered))

    # Body evidence gate: filename hints alone never decide include.
    # - exclude wins when non-geo cues dominate;
    # - include requires 2+ body geoscience terms OR ml-cues+subjects;
    # - a single body geo term (possibly just one place name) with any non-geo
    #   signal falls through to uncertain for human review.
    if all_non and len(body_geo_hits) <= 1:
        return "exclude", reason, 3
    if ml_cues and ml_subjects:
        return "include", reason, 3
    if len(body_geo_hits) >= 2:
        return "include", reason, 3
    if len(body_geo_hits) == 1 and not all_non:
        return "uncertain", reason, 3  # single place name alone
    if not body_geo_hits and name_geo_hits and not all_non:
        return "uncertain", reason, 3  # only filename hint, no body evidence
    if not body_geo_hits and all_non:
        return "exclude", reason, 3
    if text_len >= 1500:
        return "exclude", reason, 3
    return "uncertain", reason, 3


# Platform names / watermarks that regularly appear in downloaded PDFs as
# "/Author" or on the first page — they are NOT the paper's authors.
_PLATFORM_MARKERS = {
    "cnki", "cnki.net", "cnki online", "cnki.net on line",
    "z-library", "zlib", "z-lib.org", "zlib.org", "zlibrary",
    "free-docs", "free docs", "freedocs",
    "mdpi", "mdpi.com", "mdpi open access",
    "scihub", "sci-hub", "sci hub",
    "researchgate", "research gate",
    "academia.edu", "academia",
    "semantic scholar", "semanticscholar",
    "google scholar", "scholar",
    "elsevier", "springer", "wiley", "taylor & francis",
    "wikipedia", "wikimedia",
}


def _is_platform_or_watermark(token: str) -> bool:
    low = token.strip().lower()
    if not low:
        return True
    if low in _PLATFORM_MARKERS:
        return True
    # Common download/watermark markers: "uploaded by", "downloaded from",
    # "obtained from", "provided by" followed by a platform.
    if re.match(r"^(uploaded|downloaded|obtained|provided|collected|archived)\b", low):
        return True
    if low.endswith("(z-lib.org)") or low.endswith("(zlibrary)") or low.endswith("(z-lib)"):
        return True
    return False


def _clean_author_token(token: str) -> str:
    token = token.strip().strip(",;")
    # Strip superscript/footnote markers but NOT genuine author-initials dots:
    # "Zhang S.", "A. B. Smith", etc. Keep ".", "'", "-" in names.
    token = re.sub(r"[\d\*∗†‡§¶#]+", "", token).strip()
    return token.strip()


def _looks_like_individual_name(token: str) -> bool:
    """Return True only if ``token`` plausibly represents a human author name.

    Chinese names are 2–4 Hanzi characters. English names are 1–3 words each
    starting with an uppercase letter. Anything else (institution, platform,
    download watermark, all-lowercase string, a plain year) returns False so
    that caller code leaves the field empty instead of guessing.
    """
    if not token:
        return False
    if _is_platform_or_watermark(token):
        return False
    # All-lowercase / all-upper without capitalisation → not a personal name.
    if token.islower() or (token.isupper() and re.search(r"[a-z]", token)):
        return False
    if re.fullmatch(r"\d{4}", token.strip()):
        return False  # plain year, e.g. "Received 2019" parsed as a token
    if re.search(r"[\u3400-\u9fff]", token):
        stripped = re.sub(r"[\s\-\.·]+", "", token)
        return 2 <= len(stripped) <= 6 and not re.search(r"[\d\*∗†‡§¶#]", stripped)
    parts = token.split()
    if not (1 <= len(parts) <= 3):
        return False
    # Each part must be capitalised-letter or a single initial like "S." or "A.B.".
    for p in parts:
        if not p:
            return False
        if not re.fullmatch(r"[A-Z][A-Za-z'’.\-]+", p):
            # Allow single-letter initials like "S" or "A.B."
            if not re.fullmatch(r"[A-Z]\.?", p):
                return False
    return True


def _looks_like_institution(token: str) -> bool:
    """A token that is definitely NOT an individual name but still plausible
    as a collective author (academy, university, institute, lab ...).

    We keep this conservative: only recognisable organisation words count.
    CNKI / Z-Library etc. are platforms, not institutions — they return False
    here and are then filtered out entirely.
    """
    if not token or _is_platform_or_watermark(token):
        return False
    low = token.lower()
    org_words = (
        "university", "college", "academy", "institute", "lab", "laboratory",
        "center", "centre", "department", "school", "consortium", "committee",
        "society", "corporation", "inc", "llc",
    )
    return any(w in low for w in org_words)


def guess_authors(
    pdf_author_raw: str | None, first_text: str
) -> tuple[list[str], int | None, str | None]:
    """Best-effort author extraction; never returns a platform/watermark.

    Returns ``(authors, evidence_page, kind)``. ``kind`` is ``"individual"``,
    ``"institution"`` or ``None`` (confidence too low). Callers MUST NOT store
    these as *verified* bibliography fields — they are heuristic candidates
    awaiting human science review.
    """
    # Pre-compute the author-candidate lines from the first pages; both the
    # separator-based path and the whitespace-fallback path use them.
    lines = [ln.strip() for ln in first_text.split("\n") if ln.strip()][:20]

    pieces: list[str] = []
    if isinstance(pdf_author_raw, str) and pdf_author_raw.strip():
        pieces = re.split(r"\s*;|\s+and\s+|,\s*(?=[A-Z])", pdf_author_raw.strip())

    if not pieces:
        # Prefer lines with explicit separators first (comma / & / "and"); many
        # PDFs (especially Chinese) list names without punctuation, so the
        # fallback below catches whitespace-separated author lines.
        for line in lines:
            if len(line) > 200 or "@" in line:
                continue
            if not re.search(r",|&|\band\b", line):
                continue
            pieces = re.split(r"\s+and\s+|&|,", line)
            if len(pieces) >= 2:
                break
            pieces = []

    cleaned = [_clean_author_token(p) for p in pieces]
    cleaned = [p for p in cleaned if p and len(p) >= 2 and not _is_platform_or_watermark(p)]
    individuals = [p for p in cleaned if _looks_like_individual_name(p)]
    institutions = [p for p in cleaned if _looks_like_institution(p)]

    if individuals:
        return individuals[:12], 1, "individual"
    if institutions:
        return institutions[:6], 1, "institution"

    # Fallback: whitespace-separated line — e.g. "Zhang S. Li X." or Chinese
    # names on one line without commas. Only accept 2+ plausible personal
    # names; otherwise leave the field empty rather than guessing.
    _HEADER_WORDS = frozenset(
        "received accepted submitted revised published publication "
        "draft final in for on doi abstract introduction keywords "
        "corresponding author corresponding authors corresponding "
        "author authors authorship".split()
    )
    for line in lines:
        if len(line) > 200 or "@" in line:
            continue
        tokens = re.split(r"\s+", line.strip())
        tokens = [_clean_author_token(t) for t in tokens]
        tokens = [t for t in tokens if t and len(t) >= 2 and not _is_platform_or_watermark(t)]
        # Drop line headers ("Received", "Abstract" ...) that are never names.
        tokens = [
            t for t in tokens
            if t.lower() not in _HEADER_WORDS and not re.fullmatch(r"\d{4}", t.strip())
        ]
        names = [t for t in tokens if _looks_like_individual_name(t)]
        if len(names) >= 2:
            return names[:12], 1, "individual"
    return [], None, None


_YEAR_TAG_PATTERNS = (
    # Publication / issue year — prefer these over received/accepted.
    ("published", re.compile(r"(?:published|publication)\b[^0-9]{0,40}((?:19|20)\d{2})", re.I)),
    ("in", re.compile(r"(\b(?:19|20)\d{2}\b)\s*[.,]?\s*in\s", re.I)),
    ("copyright", re.compile(r"©\s*((?:19|20)\d{2})")),
    ("volume", re.compile(r"\b(?:Volume|Vol\.?|Number|No\.?)\s*\d+\D{0,30}((?:19|20)\d{2})", re.I)),
    ("doi_year", re.compile(r"10\.\d{4,9}/[^\s]*?(\d{4})(?:$|[^0-9])", re.I)),
    # Received / accepted / revised — always capture but report separately so the
    # caller can decide whether to use them. They are NOT the publication year.
    ("received", re.compile(r"(?:received|receipt|submitted)\b[^0-9]{0,40}((?:19|20)\d{2})", re.I)),
    ("accepted", re.compile(r"(?:accepted)\b[^0-9]{0,40}((?:19|20)\d{2})", re.I)),
    ("revised", re.compile(r"(?:revised|revision|rev\.)\b[^0-9]{0,40}((?:19|20)\d{2})", re.I)),
)


def guess_year(first_text: str) -> tuple[int | None, dict[str, int]]:
    """Heuristic year extraction; returns ``(publication_year, tagged_map)``.

    ``tagged_map`` records each hit's tag (``received`` / ``accepted`` /
    ``revised`` / ``published`` / ...)  — only ``published`` / ``copyright`` /
    ``volume`` / ``doi_year`` hits are candidates for the publication year.
    Callers MUST NOT mix received/accepted/revised with the publication year.
    """
    head = first_text[:1500]
    tagged: dict[str, int] = {}
    for tag, pattern in _YEAR_TAG_PATTERNS:
        m = pattern.search(head)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2030 and tag not in tagged:
                tagged[tag] = year

    pub_tags = ("published", "copyright", "volume", "doi_year", "in")
    pub_year = None
    for tag in pub_tags:
        if tagged.get(tag):
            pub_year = tagged[tag]
            break
    if pub_year is None:
        # Fall back to the largest plausible year found in the first 400 chars
        # if the header patterns matched nothing — still not reliable, so the
        # caller must mark it as "pending user science review".
        years = [int(m.group(0)) for m in YEAR_RE.finditer(head[:400])]
        plausible = [y for y in years if 1950 <= y <= 2030]
        if plausible:
            pub_year = max(plausible)
    return pub_year, tagged


def main() -> None:
    if not PDF_ROOT.is_dir():
        raise SystemExit(f"missing PDF root: {PDF_ROOT}")
    pdfs = sorted(PDF_ROOT.rglob("*.pdf"), key=lambda p: str(p).lower())
    records = []
    by_sha: dict[str, list[str]] = defaultdict(list)

    for idx, pdf in enumerate(pdfs, start=1):
        rel = pdf.relative_to(ROOT).as_posix()
        digest = sha256_file(pdf)
        hints = filename_hints(pdf.stem)
        biblio = first_page_biblio(pdf)
        by_sha[digest].append(rel)
        pending = []
        title = biblio["pdf_title"]
        if title and len(title) >= 8 and not title.lower().startswith("untitled"):
            title_source = "pdf_metadata"
            title_is_filename = False
            pending.append("title from PDF metadata; not yet user-verified")
        else:
            title = hints["title_from_filename"]
            title_source = "filename"
            title_is_filename = True
            pending.append("title is filename placeholder")

        authors: list[str] = []
        authors_source = None
        if biblio["pdf_author"]:
            raw = re.split(r"\s*;|\s* and \s*|,\s*", biblio["pdf_author"])
            cleaned = [a.strip() for a in raw if a.strip() and len(a.strip()) <= 80]
            if cleaned:
                authors = cleaned[:12]
                authors_source = "pdf_metadata"
                pending.append("authors from PDF metadata; not yet user-verified")
        if not authors and hints["authors_from_filename"]:
            authors = hints["authors_from_filename"]
            authors_source = "filename_hint"
            pending.append("authors only hinted by filename; pending user science review")

        year = biblio["year_in_text"] or hints["year_from_filename"]
        if year is not None:
            pending.append("year inferred from file text/filename; pending user science review")
        doi = biblio["doi_in_text"]
        if doi:
            pending.append("DOI substring found in first pages; pending confirmation")
        if not biblio["extractable"]:
            pending.append("first pages have little extractable text; possible scan/complex layout")

        scan = _scan_text(pdf)
        decision, decision_reason, inspected_pages = classify_geoscience(scan, pdf.name)
        if not scan and not biblio["extractable"]:
            decision = "uncertain"
            decision_reason = (decision_reason + "；首页无可抽取文本，需人工核对").strip("；")

        first_text = biblio["first_pages_text"]
        # Heuristic candidates — NEVER treated as human-verified.
        candidate_authors, author_page, author_kind = guess_authors(
            biblio["pdf_author"], first_text
        )
        candidate_year, year_tags = guess_year(first_text)
        # Only title from PDF metadata (non-filename placeholder) and DOI substring
        # are heuristically trustworthy enough to pre-mark as "tentative verified".
        # Authors / year always stay pending until human science review — even if
        # the guesser returned something plausible, because we cannot distinguish a
        # genuine authored paper from a CNKI/Z-Library watermark on a downloaded
        # PDF, nor a received-year header from the publication year.
        bib_verified: list[str] = []
        if not title_is_filename and title:
            bib_verified.append("title")
        if doi:
            bib_verified.append("doi")
        bib_pages = sorted(
            {
                p
                for p in (
                    1 if (not title_is_filename and title) else None,
                    author_page,
                    1 if verified_year is not None else None,
                    1 if doi else None,
                )
                if p
            }
        )

        record = {
            "path": rel,
            "filename": pdf.name,
            "sha256": digest,
            "bytes": pdf.stat().st_size,
            "visibility": "private",
            "title": title,
            "title_source": title_source,
            "title_is_filename": title_is_filename,
            "authors": authors,
            "authors_source": authors_source,
            "year": year,
            "doi_or_url": None if not doi else f"https://doi.org/{doi}",
            "license": None,
            "license_url": None,
            "download_date": None,
            "pdf_page_count": biblio["pdf_page_count"],
            "extractable_first_pages": biblio["extractable"],
            "first_pages_chars": biblio["first_pages_chars"],
            "doi_in_first_pages": doi,
            "year_in_first_pages": biblio["year_in_text"],
            "filename_hints": hints,
            "duplicate_of": None,
            "duplicate_group": None,
            "pending_verification": pending,
            "notes": biblio["extract_note"],
            "trial_subset": False,
            "decision": decision,
            "decision_reason": decision_reason,
            "inspected_pdf_pages": inspected_pages,
            # Heuristic-only candidates. Never promoted to verified automatically —
            # human science review must confirm before they enter the verified set.
            "candidate_authors": candidate_authors,
            "candidate_authors_source": "first_pages_text" if candidate_authors else None,
            "candidate_authors_kind": author_kind,
            "candidate_authors_page": author_page,
            "candidate_year": candidate_year,
            "candidate_year_source": "header_pattern" if candidate_year else None,
            "candidate_year_tags": year_tags,
            "bibliography_verified_fields": bib_verified,
            "bibliography_evidence_pages": bib_pages,
        }
        records.append(record)
        if idx % 10 == 0 or idx == len(pdfs):
            print(f"scanned {idx}/{len(pdfs)}", flush=True)

    duplicate_groups = []
    for digest, paths in by_sha.items():
        if len(paths) > 1:
            group_id = f"dup_{digest[:12]}"
            duplicate_groups.append({"sha256": digest, "group_id": group_id, "paths": paths})
            for rec in records:
                if rec["sha256"] == digest:
                    rec["duplicate_group"] = group_id
                    others = [p for p in paths if p != rec["path"]]
                    rec["duplicate_of"] = others[0] if others else None

    stem_map: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        key = re.sub(r"[\s_\-—–.·,;:（）()【】\[\]]+", "", Path(rec["filename"]).stem.lower())
        stem_map[key].append(rec["path"])
    near_dupes = []
    for key, paths in stem_map.items():
        if len(paths) > 1:
            shas = {r["sha256"] for r in records if r["path"] in paths}
            if len(shas) > 1:
                near_dupes.append(
                    {"normalized_stem": key, "paths": paths, "note": "similar filenames, different bytes"}
                )

    entries = []
    for rec in records:
        entries.append(
            {
                "relative_path": rec["path"],
                "sha256": rec["sha256"],
                "duplicate_of": rec["duplicate_of"],
                "decision": rec["decision"],
                "reason": rec["decision_reason"],
                "inspected_pdf_pages": rec["inspected_pdf_pages"],
                "title": rec["title"],
                "title_is_filename": rec["title_is_filename"],
                # Source-level authors/year intentionally left empty here:
                # heuristic guesses (CNKI watermark, received-years ...) never
                # become verified bibliographic fields by themselves; human
                # science review must confirm first. See ``candidate_*`` below.
                "authors": [],
                "year": None,
                "doi_or_url": rec["doi_or_url"],
                "bibliography_verified_fields": rec["bibliography_verified_fields"],
                "bibliography_evidence_pages": rec["bibliography_evidence_pages"],
                "visibility": "private",
                "filename_hints": rec["filename_hints"],
                # Heuristic candidates — for the human review UI / later
                # confirmation. Import does not read these.
                "candidate_authors": rec["candidate_authors"],
                "candidate_authors_kind": rec["candidate_authors_kind"],
                "candidate_authors_page": rec["candidate_authors_page"],
                "candidate_year": rec["candidate_year"],
                "candidate_year_tags": rec["candidate_year_tags"],
            }
        )
    counts = {
        "include": sum(1 for e in entries if e["decision"] == "include"),
        "exclude": sum(1 for e in entries if e["decision"] == "exclude"),
        "uncertain": sum(1 for e in entries if e["decision"] == "uncertain"),
    }
    manifest = {
        "contract_version": "0.2.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pdf_root": "PDF/",
        "visibility_default": "private",
        "classification_method": (
            "content scan of title + first 3 pages against geoscience / ML-for-geoscience / "
            "non-geoscience vocabularies; filename words are only candidate signals and never "
            "decide eligibility on their own. Unknown bibliographic fields stay empty."
        ),
        "counts": counts,
        "entries": entries,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {MANIFEST_PATH} (include={counts['include']} exclude={counts['exclude']} "
        f"uncertain={counts['uncertain']})"
    )

    payload = {
        "contract_version": "0.2.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pdf_root": "PDF/",
        "visibility_default": "private",
        "notes": (
            "Local private corpus inventory for Wave 0. All PDFs are private and not redistributable "
            "without explicit user confirmation. Unknown bibliographic fields are null. Values marked "
            "pending_verification require user science review and must not be treated as locked metadata."
        ),
        "total_files": len(records),
        "unique_sha256": len(by_sha),
        "duplicate_groups": duplicate_groups,
        "similar_filename_groups": near_dupes,
        "sources": records,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT_PATH} ({len(records)} files, {len(by_sha)} unique hashes, "
        f"{len(duplicate_groups)} duplicate groups)"
    )


if __name__ == "__main__":
    main()
