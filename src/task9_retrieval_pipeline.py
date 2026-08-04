"""
Task 9 â€” Retrieval Pipeline HoÃ n Chá»‰nh.

Káº¿t há»£p semantic search + lexical search + reranking + PageIndex fallback
thÃ nh má»™t pipeline thá»‘ng nháº¥t.

Logic:
    1. Cháº¡y semantic_search + lexical_search song song
    2. Merge káº¿t quáº£ (RRF hoáº·c weighted fusion)
    3. Rerank
    4. Náº¿u top result score < threshold â†’ fallback sang PageIndex
    5. Return top_k results

âš ï¸ BáºªY THÆ¯á»œNG Gáº¶P â€” Ä‘á»c ká»¹ trÆ°á»›c khi code:
    Náº¿u báº¡n dÃ¹ng Ä‘iá»ƒm RRF Ä‘Ã£ fuse (Task 7) Ä‘á»ƒ so vá»›i score_threshold, báº¡n sáº½ gáº·p bug
    tháº­t: RRF max score luÃ´n â‰ˆ 1/(k+1) â‰ˆ 0.0164 (k=60) Báº¤T Ká»‚ ná»™i dung cÃ³ liÃªn quan
    hay khÃ´ng. Náº¿u Ä‘áº·t threshold tháº¥p (nhÆ° 0.005) Ä‘á»ƒ "há»£p" vá»›i thang Ä‘iá»ƒm RRF, thá»±c
    cháº¥t KHÃ”NG cÃ¢u há»i nÃ o Ä‘á»§ tháº¥p Ä‘á»ƒ trigger fallback ná»¯a â€” ká»ƒ cáº£ query hoÃ n toÃ n vÃ´
    nghÄ©a váº«n tráº£ vá» káº¿t quáº£ "hybrid" (rÃ¡c) thay vÃ¬ fallback Ä‘Ãºng nhÆ° thiáº¿t káº¿.

    CÃ¡ch sá»­a Ä‘Ãºng: giá»¯ Ä‘iá»ƒm cosine similarity Gá»C cá»§a semantic_search (trÆ°á»›c khi qua
    RRF) lÃ m cÄƒn cá»© quyáº¿t Ä‘á»‹nh fallback, tÃ¡ch biá»‡t khá»i Ä‘iá»ƒm RRF dÃ¹ng Ä‘á»ƒ sáº¯p xáº¿p káº¿t
    quáº£ cuá»‘i cÃ¹ng. Calibrate threshold báº±ng cÃ¡ch tá»± Ä‘o: cháº¡y vÃ i cÃ¢u há»i cháº¯c cháº¯n
    liÃªn quan vÃ  vÃ i cÃ¢u cháº¯c cháº¯n láº¡c Ä‘á»/rÃ¡c qua semantic_search, xem khoáº£ng cÃ¡ch
    Ä‘iá»ƒm sá»‘ giá»¯a hai nhÃ³m rá»“i chá»n ngÆ°á»¡ng náº±m giá»¯a.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold nÃ y báº±ng cÃ¡ch tá»± Ä‘o Ä‘iá»ƒm cosine cá»§a semantic_search
# cho cÃ¢u há»i liÃªn quan vs cÃ¢u há»i láº¡c Ä‘á» (xem ghi chÃº á»Ÿ trÃªn) â€” Äá»ªNG copy nguyÃªn
# giÃ¡ trá»‹ máº«u, má»—i corpus/embedding model sáº½ cho khoáº£ng Ä‘iá»ƒm khÃ¡c nhau.
SCORE_THRESHOLD = 0.3   # Náº¿u best score (cosine gá»‘c) < threshold â†’ fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


from concurrent.futures import ThreadPoolExecutor


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoÃ n chá»‰nh vá»›i fallback logic.

    Pipeline:
        Query
          â”œâ†’ Semantic Search â†’ dense_results (giá»¯ Ä‘iá»ƒm cosine gá»‘c)
          â”œâ†’ Lexical Search  â†’ sparse_results
          â”‚
          â”œâ†’ Merge (RRF) â†’ merged_results
          â”œâ†’ Rerank â†’ reranked_results
          â”‚
          â””â†’ If dense_results[0]["score"] < threshold:
                â””â†’ PageIndex Vectorless â†’ fallback_results

    Args:
        query: CÃ¢u truy váº¥n
        top_k: Sá»‘ lÆ°á»£ng káº¿t quáº£ cuá»‘i cÃ¹ng
        score_threshold: NgÆ°á»¡ng Ä‘iá»ƒm cosine gá»‘c tá»‘i thiá»ƒu (KHÃ”NG pháº£i Ä‘iá»ƒm RRF)
        use_reranking: CÃ³ Ã¡p dá»¥ng reranking hay khÃ´ng

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoáº·c 'pageindex'
        }
    """
    fetch_k = top_k * 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(semantic_search, query, fetch_k)
        future_sparse = executor.submit(lexical_search, query, fetch_k)
        dense_results = future_dense.result() or []
        sparse_results = future_sparse.result() or []

    # If the dense index is unavailable, do not discard useful BM25 evidence.
    # PageIndex is a structural fallback, while BM25 remains a better first
    # fallback for direct legal terms such as "thử việc" or "Điều 25".
    if not dense_results and sparse_results:
        for item in sparse_results:
            item["source"] = "bm25_fallback"
        return rerank(query, sparse_results, top_k=top_k, method="rrf")

    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(f"  âš  Semantic best score ({best_score:.3f}) < threshold ({score_threshold}) -> Triggering PageIndex Fallback")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for item in fallback:
                item["source"] = "pageindex"
            return fallback[:top_k]

    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Thá»i gian thá»­ viá»‡c tá»‘i Ä‘a Ä‘á»‘i vá»›i ngÆ°á»i lao Ä‘á»™ng lÃ  bao lÃ¢u?",
        "Quy Ä‘á»‹nh vá» nghá»‰ háº±ng nÄƒm vÃ  thanh toÃ¡n tiá»n lÆ°Æ¡ng nghá»‰ phÃ©p?",
        "Há»£p Ä‘á»“ng lao Ä‘á»™ng gá»“m nhá»¯ng loáº¡i nÃ o theo Bá»™ Luáº­t Lao Äá»™ng 2019?",
        "xyzabc123nonsense",  # Query khÃ´ng cÃ³ káº¿t quáº£ â†’ test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")

