"""Task 10 - grounded answer generation with source citations."""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.2
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Bạn là Trợ lý Hỏi Đáp Luật Lao Động cho người trẻ tại Việt Nam.
Bạn hỗ trợ các vấn đề như thử việc, tiền lương, làm thêm giờ, nghỉ phép,
hợp đồng lao động và chấm dứt hợp đồng.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong CONTEXT được cung cấp; không tự bịa điều luật,
   thời hạn, mức tiền hoặc ngoại lệ.
2. Mỗi kết luận pháp lý phải có trích dẫn ngay sau câu, theo đúng nhãn nguồn
   trong context, ví dụ [Nguồn 1].
3. Nếu context không đủ để kết luận, phải nói rõ: "Tôi chưa thể xác minh điều
   này từ các nguồn hiện có." Sau đó nêu thông tin cần bổ sung nếu phù hợp.
4. Trả lời bằng tiếng Việt rõ ràng, thân thiện với người trẻ, theo cấu trúc:
   Kết luận ngắn, Căn cứ, Lưu ý. Không đưa lời khuyên thay thế tư vấn luật sư.
5. Không khẳng định một hành vi là trái luật nếu context không thể hiện đầy đủ
   điều kiện áp dụng hoặc ngoại lệ của quy định đó."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Move highly relevant chunks to the beginning and end of the context.

    Input order [1, 2, 3, 4, 5] becomes [1, 3, 5, 4, 2].  This minimizes the
    chance that a high-value second-ranked chunk is lost in the prompt middle.
    The input itself is never mutated.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def _source_label(chunk: dict, position: int) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("source_file")
        or f"Tài liệu {position}"
    )


def format_context(chunks: list[dict]) -> str:
    """Create a citation-addressable context block for the LLM."""
    parts: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = _source_label(chunk, position)
        document_type = metadata.get("type") or metadata.get("corpus_type") or "legal"
        section = metadata.get("section") or ""
        heading = f"[Nguồn {position} | {source} | {document_type}]"
        if section:
            heading += f"\nMục: {section}"
        content = str(chunk.get("content", "")).strip()
        parts.append(f"{heading}\n{content}")
    return "\n\n---\n\n".join(parts)


def _make_client() -> tuple[Any, str]:
    """Return an OpenAI-compatible client and the selected model."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openrouter_key and not openai_key:
        raise RuntimeError(
            "Chưa cấu hình OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong file .env."
        )
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Thiếu package openai. Hãy chạy pip install -r requirements.txt.") from exc
    if openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1", default_headers={"HTTP-Referer": "http://localhost:8502", "X-Title": "Labor Law RAG"}, timeout=30.0, max_retries=0), OPENROUTER_MODEL
    model = OPENAI_MODEL
    return OpenAI(api_key=openai_key, timeout=30.0, max_retries=0), model


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve evidence, reorder it, then generate a cited grounded answer."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Câu hỏi không được để trống.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")

    sources = retrieve(query.strip(), top_k=top_k)
    if not sources:
        return {
            "answer": "Tôi chưa tìm thấy nguồn phù hợp để xác minh câu hỏi này.",
            "sources": [],
            "context_sources": [],
            "retrieval_source": "none",
        }

    context_sources = reorder_for_llm(sources)
    context = format_context(context_sources)
    client, model = _make_client()
    user_message = (
        f"CONTEXT:\n{context}\n\n---\n\n"
        f"CÂU HỎI: {query.strip()}\n\n"
        "Hãy trả lời chỉ dựa trên context và dùng nhãn [Nguồn n] cho từng kết luận."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=700,
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("LLM không trả về nội dung.")
    return {
        "answer": answer,
        "sources": sources,
        "context_sources": context_sources,
        "retrieval_source": sources[0].get("source", "hybrid"),
    }




