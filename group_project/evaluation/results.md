# RAG Evaluation & A/B Benchmark Results

- **Run time (UTC):** 2026-08-04T05:46:34.245264+00:00
- **Golden dataset:** 20 câu hỏi luật lao động
- **Top-k:** 5
- **Execution mode:** `offline_proxy`

> **Ghi chú phương pháp:** Môi trường chạy chưa cài RAGAS/datasets, vì vậy bảng dưới là benchmark
> deterministic proxy trên toàn bộ dataset: faithfulness đo token grounding của câu trả lời trích xuất;
> answer relevance đo coverage của evidence trong câu trả lời; context recall/precision đo bằng nhãn evidence.
> Script đã có chế độ `--mode ragas` để chạy bốn metric RAGAS chính thức khi cài dependency.

## Configurations

- **Config A — `A_bm25_top5`:** BM25 lexical retrieval over chunked legal/news corpus.
- **Config B — `B_pageindex_top5`:** PageIndex-style local structural section retrieval.

## Overall Scores

| Metric | Config A | Config B | Δ (A−B) |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer Relevance | 0.479 | 0.383 | +0.096 |
| Context Recall | 0.871 | 0.988 | -0.117 |
| Context Precision | 0.650 | 0.870 | -0.220 |
| **Macro average** | 0.750 | 0.810 | -0.060 |
| Mean retrieval latency | 0.0136s | 0.0504s | -0.0368s |

## Per-question Results

| ID | Category | Difficulty | A avg | B avg | Winner |
|---|---|---|---:|---:|---|
| LL01 | hop_dong | easy | 0.738 | 0.938 | B |
| LL02 | hop_dong | easy | 0.588 | 0.650 | B |
| LL03 | hop_dong | medium | 0.817 | 0.867 | B |
| LL04 | thu_viec | medium | 0.600 | 0.762 | B |
| LL05 | thu_viec | easy | 0.833 | 0.750 | A |
| LL06 | thu_viec | medium | 0.887 | 0.938 | B |
| LL07 | dieu_chuyen | hard | 0.650 | 0.875 | B |
| LL08 | cham_dut_hop_dong | medium | 0.875 | 0.812 | A |
| LL09 | cham_dut_hop_dong | hard | 0.625 | 0.938 | B |
| LL10 | cham_dut_hop_dong | hard | 0.650 | 0.700 | B |
| LL11 | hoc_nghe | medium | 0.762 | 0.938 | B |
| LL12 | tien_luong | easy | 0.787 | 0.688 | A |
| LL13 | lam_them | medium | 0.887 | 0.787 | A |
| LL14 | lam_them | hard | 0.642 | 0.650 | B |
| LL15 | nghi_ngoi | easy | 0.683 | 0.700 | B |
| LL16 | nghi_ngoi | medium | 0.588 | 0.762 | B |
| LL17 | nghi_ngoi | medium | 0.825 | 0.938 | B |
| LL18 | nghi_ngoi | easy | 0.938 | 0.938 | Tie |
| LL19 | tien_luong | medium | 0.850 | 0.750 | A |
| LL20 | ky_luat | hard | 0.775 | 0.825 | B |

## Worst Performers (Bottom 3)

| # | ID | Question | A avg | B avg | Failure stage |
|---:|---|---|---:|---:|---|
| 1 | LL02 | Bộ luật Lao động 2019 quy định có những loại hợp đồng lao động nào? | 0.588 | 0.650 | Retrieval / chunk coverage |
| 2 | LL16 | Người lao động Việt Nam có tổng cộng bao nhiêu ngày nghỉ lễ, Tết hưởng nguyên lương trong năm? | 0.588 | 0.762 | Retrieval / chunk coverage |
| 3 | LL04 | Thử việc vị trí lập trình viên cần trình độ cao đẳng tối đa bao lâu và được trả tối thiểu bao nhiêu phần trăm lương? | 0.600 | 0.762 | Answer extraction |

## A/B Analysis

**Config B (PageIndex)** có macro average cao hơn trên bộ kiểm thử này. BM25 thường mạnh với truy vấn chứa thuật ngữ hoặc số điều cụ thể; PageIndex có lợi thế khi heading/section của tài liệu được chuyển đổi sạch.

Điểm yếu đáng chú ý là chất lượng Markdown sau chuyển đổi: một số điều luật bị chia giữa hai chunk hoặc mất dòng, làm giảm context recall dù nguồn gốc đúng.

## Recommendations

1. **Chunk theo Điều/Khoản:** tách văn bản pháp luật bằng heading `Điều n` trước, sau đó mới áp dụng giới hạn ký tự; kỳ vọng tăng Context Recall cho câu hỏi nhiều điều kiện.
2. **Bật dense retrieval:** cài ChromaDB + sentence-transformers và build lại collection; sau đó bổ sung Config C Hybrid RRF để đo đúng lợi ích semantic + lexical.
3. **Rerank bằng coverage pháp lý:** ưu tiên chunk chứa số điều, tỷ lệ, thời hạn và cùng nguồn luật; kỳ vọng tăng Context Precision và giảm context nhiễu từ tin tức.
4. **Chạy RAGAS chính thức:** dùng judge model độc lập, `python -m group_project.evaluation.eval_pipeline --mode ragas --limit 5` trước để kiểm soát rate limit, sau đó chạy đủ 20 câu khi có quota.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline --mode offline --top-k 5
# RAGAS (sau khi cài dependency đánh giá):
.\.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline --mode ragas --limit 5 --top-k 5
```
