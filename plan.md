# Plan — Hệ tìm kiếm ngữ nghĩa hai tầng (A1)

Dataset chính: `data/VieQuADRetrieval` (text-only, human-annotated, có sẵn qrels).
Mở rộng sau (nếu còn thời gian): `data/t2ranking_vi/subset_150`, ảnh/multimodal.

Xem chi tiết yêu cầu gốc ở [requirement.txt](requirement.txt).

---

## 0. Chuẩn bị dữ liệu
- [x] Fetch/inspect VieQuADRetrieval (`data/VieQuADRetrieval/`)
- [x] Fetch/inspect t2ranking_vi subset (dùng sau)
- [x] Commit `data/VieQuADRetrieval/` vào git (đã bỏ nested `.git` của bản clone HF, add như file thường)
- [x] Viết script load corpus/queries/qrels của VieQuADRetrieval thành dạng dùng được cho pipeline → [viequad_loader.py](viequad_loader.py) (`load_corpus`, `load_queries`, `load_qrels`)

## 1. Hạ tầng lưu trữ (Postgres)
- [x] Setup Postgres qua Docker (`pgvector/pgvector:pg16`) → [docker-compose.yml](docker-compose.yml), container `httt-rag-a1-db` chạy ở port 5432
- [x] Schema bảng `documents` (id, title, body, tsv generated từ `body` với config `simple`, embedding vector(768)) → [db/init.sql](db/init.sql), có index GIN (tsv) + HNSW (embedding)
- [x] Script ingest: đọc corpus VieQuADRetrieval → insert vào Postgres, tính tsvector + embedding (`intfloat/multilingual-e5-base`) → [ingest.py](ingest.py) — đã chạy, 2490/2490 documents có embedding trong DB

## 2. Tầng 1 — Retrieval
- [x] Chọn model embedding: `intfloat/multilingual-e5-base` (lý do: xem mục Ghi chú)
- [x] Encode corpus, lưu embedding vào pgvector → xong ở Phần 1 (`ingest.py`)
- [x] Query BM25 (`tsvector`) → top-N → [retrieval.py](retrieval.py) `bm25_search()`
- [x] Query dense (pgvector cosine) → top-N → `dense_search()`
- [x] Gộp 2 danh sách bằng RRF → `hybrid_search()` — đã test với query thật (qid=0), tìm được 1/2 doc relevant (miss doc còn lại ở cả 3 chế độ — case lỗi để phân tích sau)

## 3. Tầng 2 — Rerank
- [x] Chọn cross-encoder rerank: `BAAI/bge-reranker-v2-m3` → [rerank.py](rerank.py)
- [x] Service rerank chạy trên top-N của hybrid (không đặt trong DB) → [search.py](search.py) `search(mode="hybrid_rerank")`, chạy được cả 4 mode `bm25/dense/hybrid/hybrid_rerank`
- [ ] So sánh có/không rerank (số liệu định lượng — làm ở Phần 5 khi có script eval Recall/MRR/nDCG)

## 4. API + Frontend
- [x] Backend API (FastAPI): endpoint `/search?q=...&mode=bm25|dense|hybrid|hybrid_rerank&k=...` → [api.py](api.py), chạy `python -m uvicorn api:app --port 8000`
- [x] Frontend đơn giản (HTML/JS thuần, không build step) → [static/index.html](static/index.html), form nhập câu hỏi + chọn mode, hiển thị top-k passage kèm score
- [x] Test thủ công qua API (curl, cả 4 mode) — cần user tự xác nhận UI trên browser hoạt động đúng

## 5. Đánh giá & Ablation
- [x] Script eval: tính Recall@k, MRR@k, nDCG@k trên qrels VieQuADRetrieval → [eval.py](eval.py)
- [x] Chạy sample 50 query cho cả 4 mode để kiểm tra đúng/tốc độ — kết quả sơ bộ: `bm25` R@10=0.44, `dense` R@10=0.63, `hybrid` R@10=0.56, `hybrid_rerank` R@10=0.66 (finding lạ: Dense đơn > Hybrid RRF về Recall — note cho ablation)
- [x] Full eval trên 2048 query cho `bm25`/`dense`/`hybrid` (đã dừng giữa `hybrid_rerank` theo yêu cầu — sẽ chạy lại riêng mode này sau, vì đây là mode nặng GPU nhất, ~60 phút):

  | Mode | Recall@10 | MRR@10 | nDCG@10 | Thời gian (2048 query) |
  |---|---|---|---|---|
  | bm25 | 0.4336 | 0.6293 | 0.4218 | 11.3s |
  | dense | 0.5969 | 0.7216 | 0.5527 | 132.6s |
  | hybrid (RRF) | 0.5396 | 0.7861 | 0.5352 | 132.7s |
  | hybrid_rerank | _chưa chạy full_ | | | |

  Finding xác nhận trên full data (không phải nhiễu sample): **Dense đơn > Hybrid RRF về Recall** — do RRF chỉ tính theo rank, "phạt" các doc mạnh 1 phía (xem case Gandhi đã trace).
- [x] `eval.py` đã sửa để lưu JSON sau mỗi mode (không mất tiến độ nếu dừng giữa đường)
- [ ] Chạy full `hybrid_rerank` (2048 query) khi rảnh GPU → hoàn thiện bảng so sánh 4 mode
- [ ] Chọn vài case tốt + vài case xấu để phân tích lỗi cụ thể (dựa trên per-query breakdown trong eval_results.json — đã có sẵn case Gandhi làm ví dụ)

## 6. Báo cáo & Demo
- [ ] Viết báo cáo (≤8 trang, không tính reference): dataset, model + trade-off, bảng metric, ablation, error analysis
- [ ] Ghi video demo hoặc chuẩn bị app demo trực tiếp
- [ ] (Optional) Slide

## 7. Mở rộng (nếu còn thời gian)
- [ ] Thêm dataset t2ranking_vi để test generalization / thêm góc ablation theo domain
- [ ] Thêm image query (CLIP/SigLIP) nếu chọn hướng multimodal

---

## Ghi chú / quyết định đang mở
- Model embedding: `intfloat/multilingual-e5-base` (thử trước, đổi sau nếu cần)
- Model rerank: `BAAI/bge-reranker-v2-m3` (thử trước, đổi sau nếu cần)
- Stack DB: Postgres qua Docker, image `pgvector/pgvector:pg16` — **đã setup xong**, container `httt-rag-a1-db` chạy port 5432 (user `rag`/pass `rag`/db `rag`). BM25 dùng `tsvector`/GIN built-in (đỡ phải cài `pg_search`, cần compile Rust/pgrx). Nếu sau muốn BM25 "xịn" hơn, đổi sang image `paradedb/paradedb` (đã bundle sẵn pg_search + pgvector).
- Python env: venv `.venv` (Python 3.12, không dùng 3.14 hệ thống để tránh rủi ro tương thích torch/sentence-transformers) → [requirements.txt](requirements.txt)
