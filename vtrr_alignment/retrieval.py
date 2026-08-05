from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
PACKAGE_RE = re.compile(r"^The (?P<package>[A-Za-z0-9_.-]+) package could answer")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RetrievalItem:
    package: str
    text: str
    score: float


class BM25Retriever:
    def __init__(self, documents: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("RAG corpus is empty")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(tokenize(text)) for _, text in documents]
        self.doc_lengths = [sum(terms.values()) for terms in self.doc_terms]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.df: Counter[str] = Counter()
        for terms in self.doc_terms:
            self.df.update(terms.keys())

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        query_terms = Counter(tokenize(query))
        scored: list[RetrievalItem] = []
        doc_count = len(self.documents)
        for index, terms in enumerate(self.doc_terms):
            score = 0.0
            doc_len = self.doc_lengths[index] or 1
            for term, query_weight in query_terms.items():
                tf = terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (doc_count - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += query_weight * idf * (tf * (self.k1 + 1) / denom)
            if score:
                package, text = self.documents[index]
                scored.append(RetrievalItem(package=package, text=text, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def load_rag_documents(path: Path) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            text = str(value)
            match = PACKAGE_RE.search(text)
            package = match.group("package") if match else ""
            documents.append((package, text))
    return documents


@lru_cache(maxsize=8)
def cached_retriever(path: str) -> BM25Retriever:
    return BM25Retriever(load_rag_documents(Path(path)))


def find_package_alternatives(
    hallucinated_packages: list[str],
    task_description: str,
    retriever: BM25Retriever,
    top_k: int = 3,
) -> dict[str, list[RetrievalItem]]:
    """For each hallucinated package, search RAG for functionally similar valid packages."""
    alternatives: dict[str, list[RetrievalItem]] = {}
    for pkg in hallucinated_packages:
        query = f"{pkg.replace('-', ' ')} {task_description[:500]}"
        items = retriever.search(query, top_k)
        if items:
            alternatives[pkg] = items
    return alternatives


def render_alternatives_context(alternatives: dict[str, list[RetrievalItem]]) -> str:
    """Render hallucinated→alternative mapping as readable context for the repair prompt."""
    if not alternatives:
        return ""
    lines = []
    for hallucinated, items in alternatives.items():
        candidates = ", ".join(item.package for item in items if item.package)
        if candidates:
            lines.append(
                f"- '{hallucinated}' does not exist on PyPI. "
                f"Suggested valid replacements (pick the most suitable): {candidates}"
            )
    return "\n".join(lines)


def render_context(items: list[RetrievalItem]) -> str:
    if not items:
        return ""
    lines = []
    for item in items:
        package = item.package or "unknown"
        lines.append(f"- {package}: {item.text}")
    return "\n".join(lines)

