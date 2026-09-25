# Plan — Hệ tìm kiếm ngữ nghĩa hai tầng (A1)

Dataset chính: `data/VieQuADRetrieval` (text-only, human-annotated, có sẵn qrels).
Đã chốt chỉ dùng VieQuADRetrieval (đã xoá t2ranking_vi). **Phạm vi: text-only** (đề cho phép, ghi rõ trong báo cáo). Không làm ảnh/multimodal.
Cách chạy: xem [README.md](README.md). Code nằm trong package [rag/](rag/).

Xem chi tiết yêu cầu gốc ở [requirement.txt](requirement.txt).

---

## 0. Chuẩn bị dữ liệu
- [x] Fetch/inspect VieQuADRetrieval (`data/VieQuADRetrieval/`)
- [x] Commit `data/VieQuADRetrieval/` vào git (đã bỏ nested `.git` của bản clone HF, add như file thường)
- [x] Viết script load corpus/queries/qrels của VieQuADRetrieval thành dạng dùng được cho pipeline → [rag/dataset.py](rag/dataset.py) (`load_corpus`, `load_queries`, `load_qrels`; `python -m rag.dataset <qid>` để xem query + doc đúng)

- [x] Thống kê + biểu đồ dataset → [rag/dataset_stats.py](rag/dataset_stats.py) (`results/dataset_stats.json`, `results/figures/*.png`) + tab **Dataset** trên UI
- **Phát hiện quan trọng về dataset (dùng cho báo cáo + error analysis):** corpus = **554 đoạn văn ngữ cảnh + 1936 cụm câu trả lời** (cụm được cắt ra từ chính đoạn văn), 19 bài Wikipedia. **Mọi query có đúng 2 doc đúng = 1 đoạn văn + 1 cụm trả lời.** Độ trùng từ vựng query–doc: đoạn văn trung vị 69%, cụm trả lời trung vị 0% ⇒ BM25 tìm ra đoạn văn ở 92% query nhưng cụm trả lời chỉ 10%. Recall@10 ≈ 0.5 phần lớn là do "chỉ tìm được nửa đoạn văn" — cần tách metric theo loại doc khi phân tích.

## 1. Hạ tầng lưu trữ (Postgres)
- [x] Setup Postgres qua Docker (`paradedb/paradedb:0.25.10-pg18` = Postgres 18 + pg_search + pgvector) → [docker-compose.yml](docker-compose.yml), container `httt-rag-a1-db` chạy ở port 5432
- [x] Schema bảng `documents` (id, title, body, 1 cột vector cho mỗi embedder) → [db/init.sql](db/init.sql), index BM25 (pg_search) + HNSW cho mỗi cột vector
- [x] Script ingest: đọc corpus VieQuADRetrieval → insert vào Postgres, tính tsvector + embedding (`intfloat/multilingual-e5-base`) → [rag/ingest.py](rag/ingest.py) — đã chạy, 2490/2490 documents có embedding trong DB

## 2. Tầng 1 — Retrieval
- [x] Chọn model embedding: `intfloat/multilingual-e5-base` (lý do: xem mục Ghi chú)
- [x] Encode corpus, lưu embedding vào pgvector → xong ở Phần 1 (`rag/ingest.py`)
- [x] Query BM25 (`pg_search`) → top-N → [rag/retrieval.py](rag/retrieval.py) `bm25_search()`
- [x] Query dense (pgvector cosine) → top-N → `dense_search()`
- [x] Gộp 2 danh sách bằng RRF → `hybrid_search()` — đã test với query thật (qid=0), tìm được 1/2 doc relevant (miss doc còn lại ở cả 3 chế độ — case lỗi để phân tích sau)

## 3. Tầng 2 — Rerank
- [x] Chọn cross-encoder rerank: `BAAI/bge-reranker-v2-m3` → [rag/rerank.py](rag/rerank.py)
- [x] Service rerank chạy trên top-N của hybrid (không đặt trong DB) → [rag/search.py](rag/search.py) `search(mode="hybrid_rerank")`, chạy được cả 4 mode `bm25/dense/hybrid/hybrid_rerank`
- [x] So sánh có/không rerank: có hạ tầng và số trên sample 20 query; số full nằm trong mục "lượt chạy LÂU" ở Phần 5

## 4. API + Frontend
- [x] Backend API (FastAPI): endpoint `/search?q=...&mode=bm25|dense|hybrid|hybrid_rerank&k=...` → [rag/api.py](rag/api.py), chạy `python -m uvicorn rag.api:app --port 8000`
- [x] Frontend đơn giản (HTML/JS thuần, không build step) → [static/index.html](static/index.html), form nhập câu hỏi + chọn mode, hiển thị top-k passage kèm score
- [x] Test thủ công qua API (curl, cả 4 mode) — cần user tự xác nhận UI trên browser hoạt động đúng
- [x] **Đơn giản hoá UI theo đề ("frontend đơn giản"):** bỏ panel Cấu hình (N / fusion / reranker) và "So sánh theo". Chỉ giữ: ô câu hỏi, mode, top-k, nút "So sánh 4 mode", dấu ✓ theo nhãn, tab Dataset. Mỗi kết quả có nhãn loại (đoạn văn / cụm trả lời; API trả `kind`). Ablation chỉ làm bằng `rag.evaluate`, không làm trên UI. API vẫn nhận các knob để thử bằng curl.
- [x] **Viết lại UI bằng React + TypeScript + SCSS modules** (Vite) → [web/](web/). Build ra `web/dist`, FastAPI serve tại `/`; `run_all` tự build nếu chưa có (cần Node ≥ 20). Dev: `npm --prefix web run dev` (port 5173, proxy sang API). Đã xoá `static/index.html`. Đã smoke test bằng curl (UI, 4 mode, `kind`, ✓, figures).
- [ ] User review UI trên browser

## 5. Đánh giá & Ablation

### Cấu hình ablation (ĐÃ CHỐT)
**Baseline:** embedding `e5-base` · reranker `bge-reranker-v2-m3` · N = 50 · RRF ở backend · k = 10

| # | Ablation | Biến thể | Mode chạy | Câu hỏi cần trả lời |
|---|---|---|---|---|
| 1 | Bảng chính (đề bắt buộc) | BM25 · Dense · Hybrid RRF · Hybrid + Rerank | cả 4 | Mỗi tầng đóng góp bao nhiêu? |
| 2 | Embedding | MiniLM (23M) · e5-small (118M) · e5-base (278M) · bge-m3 (568M) | Dense, Hybrid | Cần model đa ngôn ngữ? Model to hơn có đáng? |
| 3 | Reranker | bge-reranker-v2-m3 (568M) · mMiniLM (118M) | Hybrid + Rerank | Reranker to có đáng latency? |
| 4 | Vị trí RRF | backend (Python) · SQL | Hybrid | Đặt logic ở đâu? (ranking giống hệt, chỉ so latency) |
| 5 | Số ứng viên N | 20 · 50 · 100 | Hybrid, Hybrid + Rerank | Cần đưa bao nhiêu docs vào rerank? |
| 6 | Cấu hình nhẹ | e5-small + bge-m3 rerank + N = 20 | Hybrid + Rerank | Cấu hình nhẹ nhất có giữ được chất lượng cuối? |

- **N** = số docs mỗi retriever lấy ra để gộp RRF, cũng là số docs đưa vào cross-encoder. **k** = số kết quả trả về. N không đổi số lượng trả về, nhưng k docs trả về được chọn trong N ứng viên đó.
- **Metric:** Recall / MRR / nDCG @10, mỗi cái 2 bộ: **nhãn gốc** (đoạn văn + cụm trả lời) và **chỉ đoạn văn**; kèm latency (retrieval / rerank).
- **Dữ liệu:** 2048 query. Trên RTX 5060 ≈ 2.7 giờ (≈ 1.7 giờ nếu bỏ N = 100).
- **Hạn chế cần ghi trong báo cáo:** cặp reranker khác nhau cả kích thước, model nền và dữ liệu huấn luyện → chỉ kết luận "model nào tốt hơn", không tách riêng được tác động của kích thước. MiniLM đóng vai trò đối chứng (chỉ học tiếng Anh).

### Trạng thái
- [x] Hạ tầng: `rag.evaluate` với `--embedder / --reranker / --fusion / --candidate-n`, 2 bộ metric, `--rescore`, lưu sau mỗi mode. Mỗi embedder 1 cột vector (`python -m rag.ingest --embedders ...`); thời gian encode + số tham số → [results/ingest_stats.json](results/ingest_stats.json).
- [x] Chạy thử **20 query** cả 6 ablation → [results/eval_sample20.json](results/eval_sample20.json). Kết quả sơ bộ (nDCG chỉ đoạn văn):
  1. BM25 0.77 < Dense 0.80 < Hybrid 0.83 < Rerank **0.92**. Theo nhãn gốc thì Dense cao nhất vì tìm được cụm trả lời (40% query).
  2. Dense / Hybrid: MiniLM 0.12 / 0.45 · e5-small 0.83 / 0.90 · e5-base 0.80 / 0.83 · bge-m3 0.78 / 0.88. MiniLM hỏng; 3 model đa ngôn ngữ ngang nhau trong mức nhiễu của 20 query; e5-small nhanh nhất (14 ms so với 66–76 ms). Encode corpus: 1.1 / 2.2 / 4.7 / 17.8 giây.
  3. bge-reranker 0.92 (1093 ms) ≫ mMiniLM 0.83 (197 ms); mMiniLM không hơn không rerank.
  4. RRF SQL = backend về ranking; 74 vs 79 ms.
  5. N = 20 / 50 / 100 cùng 0.92; latency 534 / 1093 / 1725 ms → N = 20 là đủ.
  6. e5-small + rerank + N = 20: 0.92, 545 ms/query → bằng baseline, nhanh gấp đôi.
- [ ] **Chạy full 2048 query** (9 lệnh, lưu vào `results/eval_full.json`):
  ```
  python -m rag.evaluate --modes bm25 dense hybrid hybrid_rerank
  python -m rag.evaluate --modes dense hybrid --embedder e5-small
  python -m rag.evaluate --modes dense hybrid --embedder bge-m3
  python -m rag.evaluate --modes dense hybrid --embedder minilm
  python -m rag.evaluate --modes hybrid_rerank --reranker mminilm
  python -m rag.evaluate --modes hybrid --fusion sql
  python -m rag.evaluate --modes hybrid hybrid_rerank --candidate-n 20
  python -m rag.evaluate --modes hybrid hybrid_rerank --candidate-n 100
  python -m rag.evaluate --modes hybrid_rerank --embedder e5-small --candidate-n 20
  ```
- [ ] Sau khi chạy full: chốt cấu hình mặc định cho UI (ứng viên: e5-small + bge-reranker + N = 20) trong [rag/config.py](rag/config.py).
- [ ] Sau đó: `python -m rag.dataset_stats` để cập nhật biểu đồ `found_by_kind.png`
- [ ] Phân tích lỗi: thống kê theo loại lỗi trên full data. Đã có 2 case từ 20 query:
  - **#608** "Ai là người bản địa ở Puerto Rico?": **nhãn thiếu**. Rerank xếp đoạn đúng hạng 3, nhưng 2 đoạn trên (#726, #1038) cũng trả lời đúng. BM25 trượt vì đoạn đúng viết "cư dân đầu tiên", không có chữ "bản địa". Hybrid cứu từ hạng 12–13 lên 9 nhờ RRF.
  - **#4496** "Trước khi Roosevelt tuyên thệ nhậm chức…": **rerank làm tệ đi** (hạng 1 → 2, điểm 0.877 vs 0.862). Đáp án nằm trong ngoặc "32 ngày **sau khi** Hitler…", cần suy luận trước/sau.

**Bố cục báo cáo:** Bảng 1 = bảng chính (2 bộ metric) · Bảng 2 = embedding × reranker kèm trade-off · Bảng 3 = latency theo vị trí logic và N · 3–4 case lỗi.

## 6. Báo cáo & Demo
- [x] Hướng dẫn chạy + kịch bản demo + xử lý lỗi → [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md); bật toàn bộ hệ thống bằng `scripts/run_all.ps1` / `scripts/run_all.sh`
- [ ] Viết báo cáo (≤8 trang, không tính reference): dataset, model + trade-off, bảng metric, ablation, error analysis
- [ ] Ghi video demo hoặc chuẩn bị app demo trực tiếp
- [ ] (Optional) Slide

## 7. Mở rộng
- Đã chốt text-only, không làm image query.

---

## Ghi chú / quyết định đang mở
- Model embedding: `intfloat/multilingual-e5-base` là mặc định; ablation với `multilingual-e5-small`, `BAAI/bge-m3`, `all-MiniLM-L6-v2`
- Model rerank: `BAAI/bge-reranker-v2-m3` (thử trước, đổi sau nếu cần)
- Stack DB: Postgres qua Docker — **đã setup xong**, container `httt-rag-a1-db` chạy port 5432 (user `rag`/pass `rag`/db `rag`). Đã đổi sang image `paradedb/paradedb:0.25.10-pg18` (Postgres 18 + pg_search + pgvector), volume `paradedb_data`.
- Python env: venv `.venv` (Python 3.12, không dùng 3.14 hệ thống để tránh rủi ro tương thích torch/sentence-transformers) → [requirements.txt](requirements.txt)
