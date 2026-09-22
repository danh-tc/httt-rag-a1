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
- [ ] Setup Postgres local (docker) + extension `pg_search` (hoặc dùng `tsvector` built-in) và `pgvector`
- [ ] Schema bảng `documents` (id, text, tsvector, embedding vector)
- [ ] Script ingest: đọc corpus VieQuADRetrieval → insert vào Postgres, tính tsvector + embedding

## 2. Tầng 1 — Retrieval
- [ ] Chọn model embedding text tiếng Việt (ví dụ: `bge-m3`, `intfloat/multilingual-e5-base`, hay sentence-transformers hỗ trợ VN) — ghi chú lý do chọn
- [ ] Encode corpus, lưu embedding vào pgvector
- [ ] Query BM25 (tsvector/pg_search) → top-N
- [ ] Query dense (pgvector cosine) → top-N
- [ ] Gộp 2 danh sách bằng RRF → danh sách hybrid

## 3. Tầng 2 — Rerank
- [ ] Chọn cross-encoder rerank (ví dụ `bge-reranker`, hoặc cross-encoder/ms-marco đa ngôn ngữ)
- [ ] Service rerank chạy trên top-N của hybrid (không đặt trong DB)
- [ ] So sánh có/không rerank

## 4. API + Frontend
- [ ] Backend API (FastAPI/Flask): endpoint `/search?q=...&mode=bm25|dense|hybrid|hybrid_rerank`
- [ ] Frontend đơn giản (HTML/JS hoặc Streamlit): input text, hiển thị top-k passage kết quả
- [ ] Test thủ công vài query mẫu

## 5. Đánh giá & Ablation
- [ ] Script eval: tính Recall@k, MRR, nDCG@k trên qrels VieQuADRetrieval
- [ ] Chạy eval cho 4 cấu hình: BM25 đơn / Dense đơn / Hybrid RRF / Hybrid + rerank
- [ ] Bảng so sánh kết quả
- [ ] Chọn vài case tốt + vài case xấu để phân tích lỗi cụ thể

## 6. Báo cáo & Demo
- [ ] Viết báo cáo (≤8 trang, không tính reference): dataset, model + trade-off, bảng metric, ablation, error analysis
- [ ] Ghi video demo hoặc chuẩn bị app demo trực tiếp
- [ ] (Optional) Slide

## 7. Mở rộng (nếu còn thời gian)
- [ ] Thêm dataset t2ranking_vi để test generalization / thêm góc ablation theo domain
- [ ] Thêm image query (CLIP/SigLIP) nếu chọn hướng multimodal

---

## Ghi chú / quyết định đang mở
- Model embedding: _chưa chọn_
- Model rerank: _chưa chọn_
- Stack DB: Postgres + pg_search + pgvector (theo gợi ý đề bài) — cần xác nhận cài được trên máy
