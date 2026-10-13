"""Dense-only and hybrid (BM25 + RRF) retrieval over local evidence chunks.

Retrieval modes
- ``dense_only``: cosine similarity over embedding vectors of the raw question.
- ``bm25_only``: lexical BM25 over the question plus transparent term expansion.
- ``hybrid``: reciprocal rank fusion (RRF) of the dense and BM25 rankings.

The dense index is a small in-memory cosine index. It exposes the same
``search(query_vector, top_k) -> list[(chunk_id, score)]`` contract a Chroma
collection would, so it can be swapped without touching callers. Chroma is not
installed in this offline environment (PyPI is unreachable); this is recorded
as a known limitation in docs/EVALUATION.md.

Query expansion never overwrites the user query: the original string is kept in
``RetrievalTrace.question`` and the added terms are listed in
``RetrievalTrace.expanded_terms``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .domain import EvidenceChunk, SearchHit, Source
from .providers import EmbeddingProvider, ProviderError
from .store import LocalStore

DENSE = "dense"
BM25 = "bm25"

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9#'\-/]*|\d+(?:\.\d+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are",
    "was", "were", "with", "by", "as", "at", "from", "that", "this", "it",
    "be", "been", "which", "what", "how", "does", "do", "did", "using", "used",
    "study", "paper",
}

_GEO_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "埃达克质": ("adakitic",),
    "埃达克岩": ("adakite", "adakitic"),
    "莫根通": ("mognetong", "mogetong"),
    "锆石": ("zircon",),
    "定年": ("dating", "geochronology", "u-pb"),
    "年代学": ("geochronology", "age", "dating"),
    "镁铁质": ("mafic",),
    "基性": ("mafic", "basaltic"),
    "超基性": ("ultramafic",),
    "中性": ("intermediate",),
    "酸性": ("felsic", "acidic"),
    "花岗岩": ("granite", "granitic"),
    "闪长岩": ("diorite", "dioritic"),
    "二长闪长岩": ("monzodiorite",),
    "玄武岩": ("basalt", "basaltic"),
    "辉绿岩": ("diabase", "dolerite"),
    "岩体": ("pluton", "intrusion"),
    "侵入体": ("intrusion", "intrusive"),
    "岩墙": ("dike", "dyke"),
    "部分熔融": ("partial melting",),
    "分离结晶": ("fractional crystallization",),
    "结晶分异": ("fractional crystallization",),
    "地壳混染": ("crustal contamination", "assimilation"),
    "加厚下地壳": ("thickened lower crust",),
    "下地壳": ("lower crust",),
    "地壳增厚": ("crustal thickening", "thickened crust"),
    "地幔源区": ("mantle source",),
    "地幔楔": ("mantle wedge",),
    "源区": ("source region", "source"),
    "亏损地幔": ("depleted mantle",),
    "富集地幔": ("enriched mantle",),
    "软流圈": ("asthenosphere", "asthenospheric"),
    "岩石圈": ("lithosphere", "lithospheric"),
    "地幔柱": ("mantle plume",),
    "俯冲带": ("subduction zone", "subduction"),
    "俯冲": ("subduction", "subducted"),
    "洋壳": ("oceanic crust",),
    "大陆地壳": ("continental crust",),
    "构造背景": ("tectonic setting", "tectonic environment"),
    "地球化学": ("geochemistry", "geochemical"),
    "主量元素": ("major elements", "major oxides"),
    "微量元素": ("trace elements",),
    "稀土元素": ("rare earth elements", "ree"),
    "同位素": ("isotope", "isotopic"),
    "岩浆演化": ("magma evolution", "magmatic evolution"),
    "岩浆混合": ("magma mixing",),
    "岩浆": ("magma", "magmatic"),
    "弧岩浆": ("arc magma", "arc magmatism"),
    "板片断离": ("slab breakoff", "slab detachment"),
    "拉萨地体": ("lhasa terrane", "lhasa"),
    "东天山": ("eastern tianshan", "tianshan"),

    "morb": ("mid-ocean ridge basalt",),
    "oib": ("ocean island basalt",),
    "iab": ("island arc basalt", "island-arc basalt"),
    "ree": ("rare earth elements",),
    "lile": ("large ion lithophile elements",),
    "hfse": ("high field strength elements",),
    "ttg": ("tonalite trondhjemite granodiorite",),
    "la-icp-ms": ("laser ablation inductively coupled plasma mass spectrometry",),
    "xrf": ("x-ray fluorescence",),
    "ar-ar": ("argon-argon", "40ar/39ar"),
    "sr/nd": ("sr-nd", "sr nd isotopes"),
    "lu-hf": ("lu-hf", "lutetium hafnium"),
    "εhf": ("epsilon hf", "ehf"),
    "mg#": ("mg number", "magnesium number"),
    "sr/y": ("sry", "sr y ratio"),
    # Reverse / corpus-level bridges (Chinese question ↔ English paper and back).
    "地壳厚度": ("crustal thickness", "moho depth"),
    "青藏高原": ("tibetan plateau", "tibet", "qinghai-tibet"),
    "青藏": ("tibet", "tibetan"),
    "高原": ("plateau",),
    "冈底斯": ("gangdese",),
    "喜马拉雅": ("himalaya", "himalayan"),
    "氧逸度": ("oxygen fugacity", "redox", "fO2"),
    "氧化还原": ("redox",),
    "金红石": ("rutile",),
    "地壳生长": ("crustal growth", "continental growth"),
    "岛弧": ("island arc", "arc"),
    "adakitic": ("埃达克质", "adakite"),
    "crustal thickness": ("地壳厚度",),
    "redox": ("氧化还原", "氧逸度", "oxygen fugacity"),
    "tibet": ("青藏", "tibetan plateau"),
    "gangdese": ("冈底斯",),
}


def tokenize(text: str) -> list[str]:
    """Latin word/number tokens plus CJK character unigrams and bigrams."""
    tokens: list[str] = []
    for match in _LATIN_TOKEN_RE.findall(text):
        token = match.lower().strip("-/'")
        if token and token not in _STOPWORDS and not token.isdigit():
            tokens.append(token)
    for run in _CJK_RE.findall(text):
        for index, char in enumerate(run):
            tokens.append(char)
            if index + 1 < len(run):
                tokens.append(run[index : index + 2])
    return tokens


@dataclass(frozen=True)
class ExpandedQuery:
    original: str
    expanded: str
    added_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]


def expand_query(question: str) -> ExpandedQuery:
    """Add English/abbreviation synonyms using longest-match, no overlaps.

    Longer terms are matched first; once a span is consumed, shorter terms
    that overlap it are skipped (e.g. "超基性" blocks "基性" inside it).
    """
    lowered = question.lower()
    added: list[str] = []
    matched: list[str] = []
    consumed: list[tuple[int, int]] = []
    for term, synonyms in sorted(_GEO_EXPANSIONS.items(), key=lambda kv: -len(kv[0])):
        start = 0
        while True:
            pos = lowered.find(term, start)
            if pos == -1:
                break
            end = pos + len(term)
            if not any(c_start < end and pos < c_end for c_start, c_end in consumed):
                matched.append(term)
                consumed.append((pos, end))
                for synonym in synonyms:
                    if synonym and synonym not in lowered and synonym not in added:
                        added.append(synonym)
            start = pos + 1
    if not added:
        return ExpandedQuery(question, question, (), tuple(matched))
    return ExpandedQuery(question, question + " " + " ".join(added), tuple(added), tuple(matched))


class BM25Index:
    """Okapi BM25 over a fixed document set. Pure Python, no dependencies."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        self.k1 = k1
        self.b = b
        self.chunk_ids = list(chunk_ids)
        self._term_freqs: list[Counter[str]] = []
        document_freq: Counter[str] = Counter()
        lengths: list[int] = []
        for text in texts:
            tokens = tokenize(text)
            freq = Counter(tokens)
            self._term_freqs.append(freq)
            lengths.append(len(tokens))
            for term in freq:
                document_freq[term] += 1
        self._doc_count = len(texts)
        self._lengths = lengths
        self._avg_len = (sum(lengths) / self._doc_count) if self._doc_count else 0.0
        self._idf = {
            term: math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))
            for term, df in document_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunk_ids:
            return []
        scores: list[tuple[str, float]] = []
        for chunk_id, freq, length in zip(
            self.chunk_ids, self._term_freqs, self._lengths
        ):
            score = 0.0
            for term in query_terms:
                idf = self._idf.get(term)
                if not idf:
                    continue
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]



class DenseIndex:
    """Cosine-similarity index over precomputed embedding vectors."""

    def __init__(self, chunk_ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must have the same length")
        self.chunk_ids = list(chunk_ids)
        self._vectors: list[list[float]] = []
        self.dim = 0
        for vector in vectors:
            normalized = _normalize(list(vector))
            self.dim = len(normalized)
            self._vectors.append(normalized)

    def search(self, query_vector: Sequence[float], top_k: int) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        query = _normalize(list(query_vector))
        if len(query) != self.dim:
            raise ValueError(
                f"query vector dim {len(query)} does not match index dim {self.dim}"
            )
        scores: list[tuple[str, float]] = []
        for chunk_id, vector in zip(self.chunk_ids, self._vectors):
            dot = 0.0
            for a, b in zip(vector, query):
                dot += a * b
            scores.append((chunk_id, dot))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingCache:
    """JSON-backed embedding cache keyed by chunk text hash and model name."""

    def __init__(self, path: str | Path, model: str) -> None:
        self.path = Path(path)
        self.model = model
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        if payload.get("model") != self.model:
            return
        self._data = payload.get("entries", {})

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"model": self.model, "entries": self._data}),
            encoding="utf-8",
        )
        tmp.replace(self.path)
        self._dirty = False

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def vectors_for(
        self,
        chunks: Sequence[EvidenceChunk],
        embedder: EmbeddingProvider,
        *,
        window: int = 32,
    ) -> list[list[float]]:
        missing: list[tuple[int, EvidenceChunk]] = []
        vectors: list[list[float] | None] = [None] * len(chunks)
        for index, chunk in enumerate(chunks):
            entry = self._data.get(chunk.chunk_id)
            if entry and entry.get("text_hash") == self._text_hash(chunk.text):
                vectors[index] = entry["vector"]
            else:
                missing.append((index, chunk))
        # Embed in windows and persist after each window so a flaky network can
        # resume without re-embedding the whole corpus.
        batch = max(1, window)
        for start in range(0, len(missing), batch):
            window_items = missing[start : start + batch]
            computed = embedder.embed([chunk.text for _, chunk in window_items])
            for (index, chunk), vector in zip(window_items, computed):
                vectors[index] = vector
                self._data[chunk.chunk_id] = {
                    "text_hash": self._text_hash(chunk.text),
                    "vector": vector,
                }
            self._dirty = True
            self.save()
        return [vector for vector in vectors if vector is not None]


@dataclass
class RetrievalTrace:
    question: str
    mode: str
    expanded_query: str
    expanded_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]
    rrf_k: int
    candidate_k: int
    top_k: int
    english_query: str | None = None
    dense_ranking: list[tuple[str, float]] = field(default_factory=list)
    bm25_ranking: list[tuple[str, float]] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "mode": self.mode,
            "expanded_query": self.expanded_query,
            "expanded_terms": list(self.expanded_terms),
            "matched_terms": list(self.matched_terms),
            "english_query": self.english_query,
            "rrf_k": self.rrf_k,
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "dense_ranking": [
                {"chunk_id": cid, "score": score} for cid, score in self.dense_ranking
            ],
            "bm25_ranking": [
                {"chunk_id": cid, "score": score} for cid, score in self.bm25_ranking
            ],
            "hits": [hit.to_dict() for hit in self.hits],
        }


VALID_MODES = ("dense_only", "bm25_only", "hybrid")


class Retriever:
    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        embedder: EmbeddingProvider,
        dense_index: DenseIndex | None = None,
        bm25_index: BM25Index | None = None,
        cache: EmbeddingCache | None = None,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25_index = bm25_index or BM25Index(
            [c.chunk_id for c in self.chunks], [c.text for c in self.chunks]
        )
        if dense_index is not None:
            self.dense_index = dense_index
        else:
            cache = cache or EmbeddingCache("data/index/embeddings.json", _model_name(embedder))
            vectors = cache.vectors_for(self.chunks, embedder)
            self.dense_index = DenseIndex([c.chunk_id for c in self.chunks], vectors)

    @classmethod
    def from_store(
        cls,
        store: LocalStore,
        embedder: EmbeddingProvider,
        *,
        visibility: str | None = None,
        cache_path: str | Path = "data/index/embeddings.json",
        quality_ok_only: bool = True,
        min_chunk_chars: int = 40,
        **kwargs: Any,
    ) -> "Retriever":
        chunks = store.get_chunks()
        if visibility is not None:
            allowed = {s.source_id for s in store.list_sources(visibility)}
            chunks = [c for c in chunks if c.source_id in allowed]
        if quality_ok_only:
            # Scanned/garbled/empty pages are diagnostics, not answerable evidence.
            chunks = [
                c
                for c in chunks
                if c.quality == "ok" and len(c.text.strip()) >= min_chunk_chars
            ]
        cache = EmbeddingCache(cache_path, _model_name(embedder))
        return cls(chunks, embedder=embedder, cache=cache, **kwargs)

    def retrieve(
        self, question: str, mode: str = "hybrid", top_k: int = 5
    ) -> list[SearchHit]:
        return self.retrieve_with_trace(question, mode=mode, top_k=top_k).hits

    def retrieve_with_trace(
        self, question: str, mode: str = "hybrid", top_k: int = 5
    ) -> RetrievalTrace:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        expansion = expand_query(question)
        trace = RetrievalTrace(
            question=question,
            mode=mode,
            expanded_query=expansion.expanded,
            expanded_terms=expansion.added_terms,
            matched_terms=expansion.matched_terms,
            rrf_k=self.rrf_k,
            candidate_k=self.candidate_k,
            top_k=top_k,
        )
        dense_ranking: list[tuple[str, float]] = []
        bm25_ranking: list[tuple[str, float]] = []
        if mode in ("dense_only", "hybrid"):
            query_vector = self.embedder.embed([question])[0]
            dense_ranking = self.dense_index.search(query_vector, self.candidate_k)
        if mode in ("bm25_only", "hybrid"):
            bm25_ranking = self.bm25_index.search(expansion.expanded, self.candidate_k)
        trace.dense_ranking = dense_ranking
        trace.bm25_ranking = bm25_ranking

        dense_scores = {cid: score for cid, score in dense_ranking}
        bm25_scores = {cid: score for cid, score in bm25_ranking}
        if mode == "dense_only":
            fused = [(cid, score, [DENSE]) for cid, score in dense_ranking]
        elif mode == "bm25_only":
            fused = [(cid, score, [BM25]) for cid, score in bm25_ranking]
        else:
            fused = _rrf_fuse(dense_ranking, bm25_ranking, self.rrf_k)

        hits: list[SearchHit] = []
        for rank, (chunk_id, score, retrievers) in enumerate(fused[:top_k], start=1):
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank,
                    retrievers=retrievers,
                    source_id=chunk.source_id,
                    pdf_page=chunk.pdf_page,
                    dense_score=dense_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                )
            )
        trace.hits = hits
        return trace


def _rrf_fuse(
    dense_ranking: Sequence[tuple[str, float]],
    bm25_ranking: Sequence[tuple[str, float]],
    k: int,
) -> list[tuple[str, float, list[str]]]:
    if k < 1:
        raise ValueError("rrf_k must be >= 1")
    fused: dict[str, float] = {}
    paths: dict[str, list[str]] = {}
    for ranking, name in ((dense_ranking, DENSE), (bm25_ranking, BM25)):
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
            paths.setdefault(chunk_id, []).append(name)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, score, paths[chunk_id]) for chunk_id, score in ordered]


def _model_name(embedder: Any) -> str:
    config = getattr(embedder, "config", None)
    model = getattr(config, "embedding_model", None)
    if model:
        return str(model)
    return type(embedder).__name__


def source_lookup(store: LocalStore) -> dict[str, Source]:
    return {source.source_id: source for source in store.load_sources()}