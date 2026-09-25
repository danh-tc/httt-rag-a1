# Hướng dẫn chạy hệ thống và demo

Tài liệu này hướng dẫn chạy toàn bộ hệ thống từ máy trắng đến khi mở được giao diện. Có hai cách chạy:

- **Cách nhanh:** một script lo hết (mục 2).
- **Chạy tay từng bước:** dùng khi cần debug (mục 3).

## 1. Yêu cầu trước khi chạy

| Thành phần | Windows (RTX 5090) | Linux (RTX 3090) |
|---|---|---|
| Driver NVIDIA | ≥ 570 | ≥ 525 |
| Python | 3.12 (`winget install Python.Python.3.12`) | 3.12 (`python3.12` + `python3.12-venv`) |
| Node.js | ≥ 20 (`winget install OpenJS.NodeJS.LTS`), chỉ để build UI | ≥ 20 |
| Docker | Docker Desktop (backend WSL2) | Docker Engine + plugin `docker compose`; user nằm trong group `docker` |
| Cổng trống | 5432 (Postgres), 8000 (API) | như Windows |
| Dung lượng đĩa | ~10 GB: torch ~3 GB, 2 model ~3.3 GB, image ParadeDB ~1.5 GB | như Windows |

Lần chạy đầu cần Internet để tải image Docker, torch, các package npm của UI và các model từ Hugging Face: 4 embedder (`intfloat/multilingual-e5-base`, `intfloat/multilingual-e5-small`, `BAAI/bge-m3`, `sentence-transformers/all-MiniLM-L6-v2`) và 2 reranker (`BAAI/bge-reranker-v2-m3`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). Từ lần thứ hai, mọi thứ đã được cache sẵn trên máy.

## 2. Cách nhanh

**Bước 1 — tạo venv (chỉ làm một lần):**

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1
```
```bash
# Linux
bash scripts/setup_venv_linux.sh
```

Script tự cài torch đúng bản CUDA, rồi chạy `scripts/check_gpu.py`. Setup thành công khi dòng cuối in ra `OK: environment ready`.

**Bước 2 — chạy toàn bộ hệ thống:**

```powershell
# Windows: tự bật Docker Desktop nếu chưa chạy, tự mở trình duyệt
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```
```bash
# Linux (thêm HOST=0.0.0.0 nếu chạy trên server và truy cập từ máy khác)
bash scripts/run_all.sh
```

Script chạy lần lượt 7 bước:

| Bước | Việc làm | Lần đầu | Các lần sau |
|---|---|---|---|
| 1 | Kiểm tra `.venv` | — | — |
| 2 | Kiểm tra Docker, tự bật Docker Desktop nếu cần (chỉ trên Windows) | ~1 phút | tức thì |
| 3 | `docker compose up -d`, đợi Postgres sẵn sàng | tải image ~1.5 GB | vài giây |
| 4 | So số document trong DB với corpus. Nếu lệch thì chạy `python -m rag.ingest` | tải 4 embedder ~3.9 GB + encode ~30 giây trên GPU | bỏ qua |
| 5 | Nếu chưa có biểu đồ thì chạy `python -m rag.dataset_stats` | ~5 giây | bỏ qua |
| 6 | Nếu chưa có `web/dist` thì build UI (`npm ci` + `npm run build`) | ~30 giây | bỏ qua |
| 7 | Bật API và UI tại http://localhost:8000 | tải reranker ~2.2 GB | ~10–20 giây load model |

Các tuỳ chọn:

| | Windows | Linux |
|---|---|---|
| Đổi cổng | `-Port 8080` | `PORT=8080` |
| Không mở trình duyệt | `-NoBrowser` | `NO_BROWSER=1` |
| Ép ingest lại (khi đổi model embedding) | `-Reingest` | `REINGEST=1` |
| Ép build lại UI (sau khi sửa `web/`) | `-RebuildUi` | `REBUILD_UI=1` |

**Dừng hệ thống:** nhấn `Ctrl+C` để tắt API. Postgres vẫn tiếp tục chạy; muốn tắt thì dùng `docker compose stop`. Dữ liệu nằm trong volume `paradedb_data`, không mất khi tắt.

## 3. Chạy tay từng bước

Kích hoạt venv trước: `.venv\Scripts\Activate.ps1` (Windows) hoặc `source .venv/bin/activate` (Linux).

```bash
docker compose up -d                                  # 1. DB
docker exec httt-rag-a1-db pg_isready -U rag -d rag   #    -> "accepting connections"
python -m rag.ingest                                  # 2. encode + nạp 2490 document (4 embedder)
docker exec httt-rag-a1-db psql -U rag -d rag -c "SELECT count(*) FROM documents"   # -> 2490
python -m rag.dataset_stats                           # 3. thống kê + biểu đồ -> results/
npm --prefix web ci; npm --prefix web run build      # 4. build UI -> web/dist
python -m uvicorn rag.api:app --port 8000             # 5. API + UI
```

Các lệnh hữu ích khác:

```bash
python -m rag.search "Gandhi đấu tranh bất bạo động năm nào?"   # so sánh 4 mode trên terminal
python -m rag.dataset 3                                        # xem câu hỏi #3 và các tài liệu đúng
python -m rag.evaluate --sample 20 --save results/eval_sample20.json   # đánh giá nhanh
python -m rag.evaluate                                         # đánh giá đầy đủ 2048 câu hỏi
```

Sau khi chạy xong `rag.evaluate` đầy đủ, chạy lại `python -m rag.dataset_stats` để biểu đồ `found_by_kind.png` có đủ các mode.

## 4. Sử dụng giao diện

### Tab "Tìm kiếm"
- Nhập câu hỏi, chọn mode rồi bấm **Tìm**. Có 4 mode:

  | Mode | Cách hoạt động |
  |---|---|
  | BM25 | Khớp từ khoá (`pg_search`) |
  | Dense | So khớp ngữ nghĩa bằng embedding e5 |
  | Hybrid (RRF) | Gộp kết quả BM25 và Dense |
  | Hybrid + Rerank | Lấy top-50 của Hybrid, cho cross-encoder chấm điểm và sắp xếp lại |

- Chọn số kết quả (top-5 / 10 / 20). Mỗi kết quả có nhãn loại: đoạn văn ngữ cảnh hoặc cụm câu trả lời (cụm ngắn, in nghiêng).
- **So sánh 4 mode**: chạy cùng một câu hỏi qua cả 4 mode rồi xếp thành bảng, gồm thời gian (tách retrieval màu xanh và rerank màu vàng), kết quả top-1, và hạng của các tài liệu đúng.
- UI chỉ dùng cấu hình mặc định. Các ablation (số ứng viên N, model reranker, vị trí RRF, model embedding) được đánh giá bằng `python -m rag.evaluate` trên toàn bộ 2048 câu hỏi, xem [README](../README.md#đánh-giá). API vẫn nhận các tham số `embedder`, `candidate_n`, `fusion`, `reranker` nếu muốn thử bằng curl.
- **Link mở thẳng một câu hỏi** (tiện cho demo): `http://localhost:8000/?q=<câu hỏi>&mode=dense`, thêm `&compare=1` để mở bảng so sánh 4 mode. Nút ở góc phải header chuyển giao diện sáng / tối / theo hệ thống.
- **Sửa UI:** chạy API ở cổng 8000, rồi `npm --prefix web run dev` và mở http://localhost:5173. Trang tự reload khi sửa code; các request được proxy sang API.
- **Điểm số** không so sánh được giữa các mode, vì mỗi mode dùng một thang điểm khác nhau:

  | Mode | Loại điểm |
  |---|---|
  | BM25 | Điểm BM25 |
  | Dense | Độ tương đồng cosine |
  | Hybrid | Điểm RRF, khoảng 0.01–0.03 |
  | Rerank | Logit/xác suất của cross-encoder |

- Nếu câu hỏi **trùng khớp một câu trong dataset**, giao diện đánh dấu ✓ vào những tài liệu đúng theo nhãn và báo "tìm thấy x/2 tài liệu đúng". Trong đoạn văn ngữ cảnh, cụm câu trả lời được tô vàng. Khi dùng "So sánh 4 mode", bảng có thêm cột cho biết tài liệu đúng nằm ở hạng mấy trong mỗi mode.

### Tab "Dataset" (mở trực tiếp: http://localhost:8000/#dataset)
- Các ô số liệu tổng quan và 5 biểu đồ, lấy từ `results/figures/`.
- **Duyệt câu hỏi**: lọc theo nội dung hoặc ID. Bấm vào một câu hỏi để xem đoạn văn ngữ cảnh (có tô vàng cụm câu trả lời) và cụm câu trả lời. Bấm **"Tìm với câu hỏi này"** để chạy tìm kiếm ngay và xem kết quả được đánh dấu ✓.

### Kịch bản demo gợi ý (~4 phút)
1. Mở tab **Dataset** để giới thiệu dataset: 19 bài Wikipedia, 554 đoạn văn, 1936 cụm câu trả lời. Nhấn mạnh rằng mỗi câu hỏi có 2 tài liệu đúng.
2. Chọn câu hỏi #0, bấm "Tìm với câu hỏi này", rồi bấm **So sánh 4 mode**. Chỉ ra rằng đoạn văn được tìm thấy ở cả 4 mode, còn cụm câu trả lời thì khó tìm hơn nhiều.
3. Gõ một câu hỏi tự do, diễn đạt khác với văn bản gốc, ví dụ "Người lãnh đạo phong trào bất bạo động ở Ấn Độ sinh năm nào?". So sánh BM25 với Dense để thấy tìm kiếm ngữ nghĩa bắt được câu hỏi dù không trùng từ khoá.

## 5. Xử lý lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `failed to connect to the docker API ... dockerDesktopLinuxEngine` | Docker Desktop chưa chạy. Mở Docker Desktop (hoặc dùng `run_all.ps1`, script tự bật) |
| Linux: `permission denied ... docker.sock` | Chạy `sudo usermod -aG docker $USER` rồi đăng nhập lại |
| Container vừa lên đã tắt. `docker logs httt-rag-a1-db` báo *"in 18+, these Docker images are configured to store database data in a..."* | Postgres 18 yêu cầu mount volume vào `/var/lib/postgresql`, không phải `/var/lib/postgresql/data`. `docker-compose.yml` đã đúng; nếu bạn tự sửa thì đổi lại |
| `Bind for 0.0.0.0:5432 failed: port is already allocated` | Máy đã có một Postgres khác đang chạy. Tắt nó đi, hoặc đổi port trong `docker-compose.yml` và đặt `RAG_DB_DSN` tương ứng |
| `Port 8000 is already in use` | Dùng `-Port 8080` / `PORT=8080`, hoặc tắt tiến trình đang giữ cổng |
| `check_gpu.py`: `torch.cuda.is_available() is False` | Driver quá cũ, hoặc đang cài nhầm torch bản CPU. Chạy lại script setup với `-Recreate` / `RECREATE=1` |
| `check_gpu.py`: `no kernels for sm_120` | RTX 50xx cần torch bản cu128 hoặc cu130 (cu126 không hỗ trợ). Dùng `-TorchCuda cu128` |
| API khởi động rất lâu lần đầu | Đang tải model (~3 GB). Theo dõi tiến trình trong terminal |
| UI báo "Không tìm thấy kết quả" với mọi câu hỏi | DB chưa có dữ liệu. Chạy `python -m rag.ingest`, hoặc `run_all` với `-Reingest` |
| Tab Dataset báo "Chưa có biểu đồ" | Chạy `python -m rag.dataset_stats` |
| `relation "documents" does not exist` | Volume được tạo trước khi có `db/init.sql`. Chạy `docker compose down -v` (**xoá dữ liệu DB**), sau đó `docker compose up -d` rồi `python -m rag.ingest` |
| `CUDA out of memory` khi rerank | GPU đang bị tiến trình khác chiếm. Kiểm tra bằng `nvidia-smi` |
