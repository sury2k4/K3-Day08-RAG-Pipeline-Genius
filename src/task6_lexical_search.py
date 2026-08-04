"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

CORPUS: list[dict] = []
_BM25 = None


def _tokenize(text: str) -> list[str]:
    """Tokenize consistently, retaining Vietnamese words and document numbers."""
    return re.findall(r"[\wÀ-ỹ]+", (text or "").casefold(), flags=re.UNICODE)


class _SimpleBM25:
    """Small dependency-free BM25 implementation used when rank-bm25 is absent."""
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, tokenized_corpus)) / max(len(tokenized_corpus), 1)
        document_frequency = {}
        for tokens in tokenized_corpus:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        n = len(tokenized_corpus)
        self.idf = {term: max(0.0, __import__("math").log((n - freq + 0.5) / (freq + 0.5) + 1))
                    for term, freq in document_frequency.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for document in self.corpus:
            counts = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for term in query:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list")
    tokenized = [_tokenize(item.get("content", "")) for item in corpus]
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized)
    except ImportError:
        return _SimpleBM25(tokenized)


def _load_corpus() -> list[dict]:
    from src.task4_chunking_indexing import load_documents, chunk_documents
    return chunk_documents(load_documents())


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    global CORPUS, _BM25
    if not CORPUS:
        CORPUS = _load_corpus()
    if not CORPUS:
        return []
    if _BM25 is None:
        _BM25 = build_bm25_index(CORPUS)
    scores = _BM25.get_scores(_tokenize(query))
    ranked = sorted(range(len(CORPUS)), key=lambda i: scores[i], reverse=True)
    return [{"content": CORPUS[i]["content"], "score": float(scores[i]),
             "metadata": dict(CORPUS[i].get("metadata", {}))}
            for i in ranked[:top_k] if scores[i] > 0]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
