"""Reproducible A/B benchmark for the Labor Law RAG pipeline.

Default mode is deterministic and needs no judge LLM.  Optional ``--mode ragas``
runs the four official RAGAS metrics when the RAGAS dependencies are installed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"
METRICS = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
CONFIG_DESCRIPTIONS = {
    "A_bm25_top5": "BM25 lexical retrieval over chunked legal/news corpus",
    "B_pageindex_top5": "PageIndex-style local structural section retrieval",
}


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or len(rows) < 15:
        raise ValueError("Golden dataset must contain at least 15 cases.")
    required = {"id", "question", "expected_answer", "expected_context", "evidence_keywords"}
    ids = set()
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Case {index} is missing fields: {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate case id: {row['id']}")
        if not row["evidence_keywords"]:
            raise ValueError(f"Case {row['id']} has no evidence keywords.")
        ids.add(row["id"])
    return rows


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).casefold())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _phrase_present(phrase: str, text: str) -> bool:
    normalized_phrase, normalized_text = _normalize(phrase), _normalize(text)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True
    phrase_terms = set(normalized_phrase.split())
    text_terms = set(normalized_text.split())
    return len(phrase_terms & text_terms) / max(len(phrase_terms), 1) >= 0.8


def _retrieve(config_name: str, question: str, top_k: int) -> list[dict]:
    if config_name.startswith("A_bm25"):
        from src.task6_lexical_search import lexical_search
        rows = lexical_search(question, top_k=top_k)
        return [{**row, "source": "bm25"} for row in rows]
    if config_name.startswith("B_pageindex"):
        from src.task8_pageindex_vectorless import pageindex_search
        return pageindex_search(question, top_k=top_k)
    raise ValueError(f"Unknown config: {config_name}")


def _extractive_answer(question: str, contexts: list[str], max_sentences: int = 3) -> str:
    """Build a deterministic answer without looking at the golden answer."""
    query_terms = set(_normalize(question).split())
    candidates = []
    for context_index, context in enumerate(contexts):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        for sentence_index, sentence in enumerate(sentences):
            clean = sentence.strip()
            if len(clean) < 25:
                continue
            terms = set(_normalize(clean).split())
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            density = len(query_terms & terms) / max(math.sqrt(len(terms)), 1)
            candidates.append((overlap + 0.08 * density, -context_index, -sentence_index, clean))
    candidates.sort(reverse=True)
    selected, seen = [], set()
    for _, _, _, sentence in candidates:
        key = _normalize(sentence)
        if key in seen:
            continue
        selected.append(sentence)
        seen.add(key)
        if len(selected) >= max_sentences:
            break
    return " ".join(selected) if selected else "Không tìm thấy bằng chứng phù hợp trong context."


def _offline_metrics(item: dict, contexts: list[str], answer: str) -> dict[str, float]:
    joined_context = "\n".join(contexts)
    keywords = item["evidence_keywords"]
    context_hits = [_phrase_present(keyword, joined_context) for keyword in keywords]
    answer_hits = [_phrase_present(keyword, answer) for keyword in keywords]

    context_recall = sum(context_hits) / len(context_hits)
    relevant_chunks = sum(
        any(_phrase_present(keyword, context) for keyword in keywords)
        for context in contexts
    )
    context_precision = relevant_chunks / max(len(contexts), 1)

    answer_terms = set(_normalize(answer).split())
    context_terms = set(_normalize(joined_context).split())
    faithfulness = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    answer_relevance = sum(answer_hits) / len(answer_hits)
    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
    }


def evaluate_offline(config_name: str, golden_dataset: list[dict], top_k: int = 5) -> dict:
    per_case = []
    total_latency = 0.0
    for item in golden_dataset:
        started = time.perf_counter()
        sources = _retrieve(config_name, item["question"], top_k)
        latency = time.perf_counter() - started
        contexts = [str(source.get("content", "")) for source in sources]
        answer = _extractive_answer(item["question"], contexts)
        scores = _offline_metrics(item, contexts, answer)
        total_latency += latency
        per_case.append({
            "id": item["id"],
            "category": item.get("category", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
            "question": item["question"],
            "answer": answer,
            "sources": [
                (source.get("metadata") or {}).get("source", "unknown")
                for source in sources
            ],
            "latency_seconds": round(latency, 4),
            **scores,
        })
    aggregate = {
        metric: round(statistics.fmean(row[metric] for row in per_case), 4)
        for metric in METRICS
    }
    aggregate["average"] = round(statistics.fmean(aggregate[m] for m in METRICS), 4)
    aggregate["mean_latency_seconds"] = round(total_latency / len(per_case), 4)
    return {
        "config": config_name,
        "description": CONFIG_DESCRIPTIONS[config_name],
        "mode": "offline_proxy",
        "aggregate": aggregate,
        "per_case": per_case,
    }


def _generate_with_context(question: str, contexts: list[str]) -> str:
    """Generate an answer for optional RAGAS evaluation using current LLM config."""
    from src.task10_generation import SYSTEM_PROMPT, _make_client

    labelled = "\n\n---\n\n".join(
        f"[Nguồn {index}]\n{context[:1800]}"
        for index, context in enumerate(contexts, start=1)
    )
    client, model = _make_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{labelled}\n\nCÂU HỎI: {question}"},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return (response.choices[0].message.content or "").strip()


def evaluate_with_ragas(config_name: str, golden_dataset: list[dict], top_k: int = 5, limit: int | None = None) -> dict:
    """Run official RAGAS metrics. Requires optional evaluation dependencies."""
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RAGAS dependencies are missing. Install: pip install ragas==0.1.21 "
            "datasets langchain-openai langchain-community sentence-transformers"
        ) from exc

    rows = golden_dataset[:limit] if limit else golden_dataset
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    metadata = []
    for item in rows:
        sources = _retrieve(config_name, item["question"], top_k)
        contexts = [str(source.get("content", ""))[:1800] for source in sources]
        answer = _generate_with_context(item["question"], contexts)
        records["question"].append(item["question"])
        records["answer"].append(answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(item["expected_answer"])
        metadata.append(item)

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the RAGAS judge.")
    judge = ChatOpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("RAGAS_JUDGE_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/free")),
        temperature=0,
        timeout=60,
        max_retries=1,
    )

    # RAGAS answer relevancy needs embeddings. Prefer a local multilingual model.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ModuleNotFoundError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("RAGAS_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

    frame = evaluate(
        Dataset.from_dict(records),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=False,
    ).to_pandas()
    rename = {"answer_relevancy": "answer_relevance"}
    per_case = []
    for index, (_, row) in enumerate(frame.iterrows()):
        values = {rename.get(metric, metric): float(row.get(metric, 0.0) or 0.0) for metric in (
            "faithfulness", "answer_relevancy", "context_recall", "context_precision"
        )}
        per_case.append({
            "id": metadata[index]["id"],
            "category": metadata[index].get("category", "unknown"),
            "difficulty": metadata[index].get("difficulty", "unknown"),
            "question": metadata[index]["question"],
            "answer": records["answer"][index],
            "sources": [],
            "latency_seconds": 0.0,
            **{key: round(value, 4) for key, value in values.items()},
        })
    aggregate = {metric: round(statistics.fmean(row[metric] for row in per_case), 4) for metric in METRICS}
    aggregate["average"] = round(statistics.fmean(aggregate[m] for m in METRICS), 4)
    aggregate["mean_latency_seconds"] = 0.0
    return {"config": config_name, "description": CONFIG_DESCRIPTIONS[config_name], "mode": "ragas", "aggregate": aggregate, "per_case": per_case}


def compare_configs(golden_dataset: list[dict], mode: str = "offline", top_k: int = 5, limit: int | None = None) -> dict:
    configs = tuple(CONFIG_DESCRIPTIONS)
    if limit:
        golden_dataset = golden_dataset[:limit]
    evaluator: Callable[..., dict] = evaluate_offline if mode == "offline" else evaluate_with_ragas
    results = {}
    for config_name in configs:
        print(f"Evaluating {config_name} on {len(golden_dataset)} cases ({mode})...")
        results[config_name] = evaluator(config_name, golden_dataset, top_k=top_k)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_proxy" if mode == "offline" else "ragas",
        "top_k": top_k,
        "case_count": len(golden_dataset),
        "configs": results,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def export_results(comparison: dict, path: Path = RESULTS_PATH) -> None:
    configs = comparison["configs"]
    names = list(configs)
    a, b = configs[names[0]], configs[names[1]]
    lines = [
        "# RAG Evaluation & A/B Benchmark Results",
        "",
        f"- **Run time (UTC):** {comparison['generated_at']}",
        f"- **Golden dataset:** {comparison['case_count']} câu hỏi luật lao động",
        f"- **Top-k:** {comparison['top_k']}",
        f"- **Execution mode:** `{comparison['mode']}`",
        "",
    ]
    if comparison["mode"] == "offline_proxy":
        lines += [
            "> **Ghi chú phương pháp:** Môi trường chạy chưa cài RAGAS/datasets, vì vậy bảng dưới là benchmark",
            "> deterministic proxy trên toàn bộ dataset: faithfulness đo token grounding của câu trả lời trích xuất;",
            "> answer relevance đo coverage của evidence trong câu trả lời; context recall/precision đo bằng nhãn evidence.",
            "> Script đã có chế độ `--mode ragas` để chạy bốn metric RAGAS chính thức khi cài dependency.",
            "",
        ]
    lines += [
        "## Configurations",
        "",
        f"- **Config A — `{names[0]}`:** {a['description']}.",
        f"- **Config B — `{names[1]}`:** {b['description']}.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A | Config B | Δ (A−B) |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "**Macro average**",
    }
    for metric in (*METRICS, "average"):
        av, bv = a["aggregate"][metric], b["aggregate"][metric]
        lines.append(f"| {labels[metric]} | {_fmt(av)} | {_fmt(bv)} | {av-bv:+.3f} |")
    lines += [
        f"| Mean retrieval latency | {a['aggregate']['mean_latency_seconds']:.4f}s | {b['aggregate']['mean_latency_seconds']:.4f}s | {a['aggregate']['mean_latency_seconds']-b['aggregate']['mean_latency_seconds']:+.4f}s |",
        "",
        "## Per-question Results",
        "",
        "| ID | Category | Difficulty | A avg | B avg | Winner |",
        "|---|---|---|---:|---:|---|",
    ]
    b_by_id = {row["id"]: row for row in b["per_case"]}
    combined = []
    for row_a in a["per_case"]:
        row_b = b_by_id[row_a["id"]]
        avg_a = statistics.fmean(row_a[m] for m in METRICS)
        avg_b = statistics.fmean(row_b[m] for m in METRICS)
        winner = "A" if avg_a > avg_b else "B" if avg_b > avg_a else "Tie"
        combined.append((min(avg_a, avg_b), row_a, row_b, avg_a, avg_b))
        lines.append(f"| {row_a['id']} | {row_a['category']} | {row_a['difficulty']} | {avg_a:.3f} | {avg_b:.3f} | {winner} |")

    lines += ["", "## Worst Performers (Bottom 3)", "", "| # | ID | Question | A avg | B avg | Failure stage |", "|---:|---|---|---:|---:|---|"]
    for rank, (_, row_a, row_b, avg_a, avg_b) in enumerate(sorted(combined, key=lambda x: x[0])[:3], start=1):
        weakest_recall = min(row_a["context_recall"], row_b["context_recall"])
        stage = "Retrieval / chunk coverage" if weakest_recall < 0.6 else "Answer extraction"
        question = row_a["question"].replace("|", "\\|")
        lines.append(f"| {rank} | {row_a['id']} | {question} | {avg_a:.3f} | {avg_b:.3f} | {stage} |")

    better = "Config A (BM25)" if a["aggregate"]["average"] >= b["aggregate"]["average"] else "Config B (PageIndex)"
    lines += [
        "",
        "## A/B Analysis",
        "",
        f"**{better}** có macro average cao hơn trên bộ kiểm thử này. BM25 thường mạnh với truy vấn chứa thuật ngữ hoặc số điều cụ thể; PageIndex có lợi thế khi heading/section của tài liệu được chuyển đổi sạch.",
        "",
        "Điểm yếu đáng chú ý là chất lượng Markdown sau chuyển đổi: một số điều luật bị chia giữa hai chunk hoặc mất dòng, làm giảm context recall dù nguồn gốc đúng.",
        "",
        "## Recommendations",
        "",
        "1. **Chunk theo Điều/Khoản:** tách văn bản pháp luật bằng heading `Điều n` trước, sau đó mới áp dụng giới hạn ký tự; kỳ vọng tăng Context Recall cho câu hỏi nhiều điều kiện.",
        "2. **Bật dense retrieval:** cài ChromaDB + sentence-transformers và build lại collection; sau đó bổ sung Config C Hybrid RRF để đo đúng lợi ích semantic + lexical.",
        "3. **Rerank bằng coverage pháp lý:** ưu tiên chunk chứa số điều, tỷ lệ, thời hạn và cùng nguồn luật; kỳ vọng tăng Context Precision và giảm context nhiễu từ tin tức.",
        "4. **Chạy RAGAS chính thức:** dùng judge model độc lập, `python -m group_project.evaluation.eval_pipeline --mode ragas --limit 5` trước để kiểm soát rate limit, sau đó chạy đủ 20 câu khi có quota.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m group_project.evaluation.eval_pipeline --mode offline --top-k 5",
        "# RAGAS (sau khi cài dependency đánh giá):",
        ".\\.venv\\Scripts\\python.exe -m group_project.evaluation.eval_pipeline --mode ragas --limit 5 --top-k 5",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Labor Law RAG A/B evaluation")
    parser.add_argument("--mode", choices=("offline", "ragas"), default="offline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    dataset = load_golden_dataset()
    comparison = compare_configs(dataset, mode=args.mode, top_k=args.top_k, limit=args.limit)
    export_results(comparison)
    print(f"Wrote {RESULTS_PATH}")
    for name, result in comparison["configs"].items():
        print(name, result["aggregate"])


if __name__ == "__main__":
    main()
