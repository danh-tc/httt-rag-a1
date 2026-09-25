# Hệ tìm kiếm ngữ nghĩa hai tầng — VieQuADRetrieval (A1)

Tìm đoạn văn Wikipedia tiếng Việt từ câu hỏi tự do.

- **Tầng 1 (retrieval):** BM25 (`pg_search`) và dense (`multilingual-e5-base` + `pgvector`), gộp bằng RRF.
- **Tầng 2 (rerank):** cross-encoder `bge-reranker-v2-m3`, chạy ở service layer.

Yêu cầu đề bài: [requirement.txt](requirement.txt) · Tiến độ: [plan.md](plan.md) · **Hướng dẫn chạy chi tiết + demo + xử lý lỗi: [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md)**

## Cấu trúc

```
data/VieQuADRetrieval/   corpus (554 đoạn văn + 1936 cụm câu trả lời), queries (2048), qrels — parquet
db/init.sql              schema + index BM25 / HNSW / GIN
rag/
  config.py              tham số (DSN, model, RRF_K, CANDIDATE_N...), override qua env RAG_*
  dataset.py             load corpus/queries/qrels; CLI xem query + doc đúng
  dataset_stats.py       thống kê dataset + biểu đồ → results/dataset_stats.json, results/figures/
  db.py                  kết nối Postgres
  embedding.py           bi-encoder (e5-base mặc định; e5-small, bge-m3, MiniLM cho ablation), prefix theo từng model
  retrieval.py           bm25 / dense / RRF / hybrid
  rerank.py              cross-encoder
  search.py              điểm vào chung cho các mode
  metrics.py             Recall@k, MRR@k, nDCG@k
  evaluate.py            CLI đánh giá → results/*.json
  ingest.py              CLI encode + nạp corpus vào DB (mỗi embedder 1 cột vector) → results/ingest_stats.json
  api.py                 FastAPI /search, /dataset/* + frontend
web/                     frontend React + TypeScript + SCSS modules (Vite); build ra web/dist, FastAPI serve tại /
results/                 kết quả eval (per-query), thống kê + biểu đồ dataset
scripts/                 setup venv, run_all (bật toàn bộ hệ thống), kiểm tra GPU
docs/RUN_GUIDE.md        hướng dẫn chạy, dùng UI, kịch bản demo, xử lý lỗi
```

## Chạy

**1. Tạo venv** (Python 3.12). Script tự cài torch đúng bản CUDA, rồi kiểm tra GPU bằng [scripts/check_gpu.py](scripts/check_gpu.py):

| Máy | Lệnh | torch |
|---|---|---|
| Windows + RTX 5090 (Blackwell) | `powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1` | `+cu128`, cần driver ≥ 570 |
| Linux + RTX 3090 (Ampere) | `bash scripts/setup_venv_linux.sh` | `+cu126`, cần driver ≥ 525 (đổi bằng `TORCH_CUDA=cu128`) |

Nếu `.venv` đã tồn tại, thêm `-Recreate` (Windows) hoặc `RECREATE=1` (Linux) để xoá và tạo lại.

**2. Bật toàn bộ hệ thống bằng 1 lệnh.** Script lần lượt kiểm tra/bật Docker → DB → ingest nếu DB trống → biểu đồ nếu chưa có → build UI nếu chưa có (cần Node.js ≥ 20) → API + UI, rồi mở trình duyệt:

```powershell
powershell -ExecutionPolicy Bypass -File scripts
un_all.ps1     # Windows
bash scripts/run_all.sh                                          # Linux (HOST=0.0.0.0 để truy cập từ máy khác)
```

Hoặc chạy tay từng bước:

```bash
docker compose up -d                          # Postgres 18 + pg_search + pgvector, port 5432
python -m rag.ingest                          # encode + nạp 2490 tài liệu bằng cả 4 embedder (~30 giây với GPU)
python -m rag.dataset_stats                   # thống kê + biểu đồ dataset

npm --prefix web ci && npm --prefix web run build   # build UI → web/dist
python -m uvicorn rag.api:app --port 8000     # mở http://localhost:8000 (tab Dataset: /#dataset)
python -m rag.search "Gandhi đấu tranh bất bạo động năm nào?"   # so sánh 4 mode trên terminal
```

## Các mode

| Mode | Mô tả |
|---|---|
| `bm25` | BM25 trên `body` (index `pg_search`) |
| `dense` | cosine trên embedding e5 (HNSW, `ef_search=200`) |
| `hybrid` | top-N BM25 + top-N dense (N=50), gộp bằng RRF (k=60) |
| `hybrid_rerank` | top-N hybrid → cross-encoder → top-k |

## Đánh giá

```bash
python -m rag.evaluate                                         # 4 mode, 2048 query → results/eval_full.json
python -m rag.evaluate --sample 20 --save results/eval_sample20.json
python -m rag.evaluate --modes hybrid hybrid_rerank --candidate-n 20      # ablation số ứng viên N
python -m rag.evaluate --modes hybrid_rerank --reranker mminilm           # ablation model reranker
python -m rag.evaluate --modes dense hybrid --embedder minilm             # ablation model embedding
python -m rag.evaluate --modes hybrid --fusion sql                        # RRF trong SQL vs backend
python -m rag.evaluate --rescore                               # tính lại metric từ ranking đã lưu, không chạy search
python -m rag.dataset 3                                        # xem query 3 và các doc được gán nhãn đúng
```

Kết quả được ghép vào file JSON có sẵn. Mỗi cặp (mode, cấu hình) được lưu dưới một khoá riêng, ví dụ `hybrid_rerank` hay `hybrid_rerank[candidate_n=20]`, nên có thể chạy từng ablation riêng mà không ghi đè lên nhau. Có 2 bộ metric:

- **Nhãn gốc** (`recall`, `mrr`, `ndcg`): cả 2 tài liệu đúng (đoạn văn + cụm câu trả lời) đều được tính. Đây là bộ chuẩn của dataset.
- **Chỉ đoạn văn** (`recall_para`, `mrr_para`, `ndcg_para`): chỉ đoạn văn được tính là đúng. Bộ này sát với nhu cầu người dùng hơn. Cụm câu trả lời gần như không trùng từ với câu hỏi, nên nó kéo lệch bộ nhãn gốc về phía dense (xem mục Dataset).

Ngoài ra, file kết quả còn có `found_paragraph` và `found_span` (tỷ lệ tìm thấy từng loại tài liệu đúng), cùng `retrieval_ms`/`rerank_ms` trên mỗi query.

Các tham số ablation (mặc định ở [rag/config.py](rag/config.py)):

| Tham số | Giá trị | Áp dụng cho |
|---|---|---|
| `--candidate-n` | số ứng viên lấy từ mỗi retriever cho RRF, cũng là số ứng viên đưa vào rerank (mặc định 50) | hybrid, hybrid_rerank |
| `--fusion` | `backend` (Python) / `sql` (một câu truy vấn Postgres); thứ hạng giống hệt nhau | hybrid, hybrid_rerank |
| `--reranker` | `bge-m3` (568M tham số) / `mminilm` (118M) | hybrid_rerank |
| `--embedder` | `e5-base` (278M, 768 chiều) / `e5-small` (118M, 384) / `bge-m3` (568M, 1024) / `minilm` (23M, 384, chỉ học tiếng Anh) | dense, hybrid, hybrid_rerank |

## Dataset

`python -m rag.dataset_stats` in bảng thống kê và lưu biểu đồ vào `results/figures/`, để dùng trong báo cáo và tab Dataset. Điểm quan trọng nhất của dataset: mỗi câu hỏi có đúng 2 tài liệu đúng, gồm **đoạn văn ngữ cảnh** và **cụm câu trả lời** (một cụm ngắn, trung vị 6 âm tiết, cắt ra từ chính đoạn văn). Hai loại tài liệu này có độ trùng từ vựng với câu hỏi rất khác nhau (trung vị 69% so với 0%): BM25 tìm ra đoạn văn ở 92% câu hỏi, nhưng chỉ tìm ra cụm câu trả lời ở 10%.
