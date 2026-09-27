"""Architecture diagram for the report (Hình 4): the full hybrid_rerank request path.

    python scripts/draw_architecture.py   # -> results/figures/architecture.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "architecture.png"

SURFACE, TEXT, TEXT_2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BOX = "#ffffff"
TIER1_FILL, TIER1_EDGE = "#eaf2fc", "#86b6ef"  # light steps of the blue ramp
TIER2_FILL, TIER2_EDGE = "#f1effb", "#9085e9"  # light steps of the violet ramp
DB_FILL, DB_EDGE = "#f3f2ee", "#c3c2b7"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9.6, 9.6))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(-9, 100)
ax.set_ylim(10, 116)
ax.axis("off")


def region(x, y, w, h, fill, edge, label, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
                                facecolor=fill, edgecolor=edge, linewidth=1.3, linestyle="--" if dashed else "-"))
    ax.text(x + 1.8, y + h - 1.6, label, ha="left", va="top", fontsize=9.5, fontweight="bold", color=TEXT_2)


def box(cx, cy, w, h, title, sub="", edge=LINE, dashed=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=BOX, edgecolor=edge, linewidth=1.4, linestyle="--" if dashed else "-"))
    if sub:
        ax.text(cx, cy + 1.6, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT)
        ax.text(cx, cy - 2.0, sub, ha="center", va="center", fontsize=8.8, color=TEXT_2, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT)


def arrow(x1, y1, x2, y2, label="", label_side="right", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=LINE,
                                 linewidth=1.4, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        dx = 1.5 if label_side == "right" else -1.5
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=8.6, color=TEXT_2, style="italic")


# --- Frontend -------------------------------------------------------------------------------------
box(42, 108, 44, 9, "Frontend (React)", "Người dùng nhập câu hỏi, chọn mode")
arrow(42, 103.5, 42, 96.5, "GET /search?q=…&mode=hybrid_rerank")

# --- Backend: query encoding ----------------------------------------------------------------------
region(4, 84, 76, 13, "#fbfaf8", DB_EDGE, "Backend: FastAPI")
box(42, 88.5, 44, 7.5, "Mã hóa câu hỏi → vector", "e5-base (GPU), dùng cho nhánh ngữ nghĩa")

# --- Tier 1: PostgreSQL + RRF ---------------------------------------------------------------------
region(4, 36, 76, 44.5, TIER1_FILL, TIER1_EDGE, "TẦNG 1 · Truy xuất ứng viên")
region(8, 54, 68, 21, DB_FILL, DB_EDGE, "PostgreSQL (ParadeDB)")
box(24, 63, 26, 11, "BM25 index", "pg_search · khớp từ khóa\ncủa câu hỏi → top-50")
box(60, 63, 26, 11, "HNSW index", "pgvector · so vector\ncâu hỏi → top-50")
# Vertical, and right of the region titles so they cross no text.
arrow(34, 84.8, 34, 68.6)
arrow(56, 84.8, 56, 68.6)
box(42, 43.5, 44, 10, "Gộp RRF", "điểm = Σ 1 / (60 + hạng)  →  50 ứng viên", edge=TIER1_EDGE, dashed=True)
arrow(24, 57.4, 36, 48.6)
arrow(60, 57.4, 48, 48.6)
ax.text(82, 43.5, "Chạy ở backend\n(Python) hoặc\ntrong SQL\n(1 truy vấn)", ha="left", va="center",
        fontsize=8.6, color=TEXT_2, linespacing=1.35)
ax.plot([80.5, 64.5], [43.5, 43.5], color=TIER1_EDGE, linewidth=1, linestyle=":")

# --- Tier 2: rerank -------------------------------------------------------------------------------
arrow(42, 38.5, 42, 31.5, "50 ứng viên + nội dung")
region(4, 13, 76, 19, TIER2_FILL, TIER2_EDGE, "TẦNG 2 · Xếp hạng lại")
box(42, 20.5, 44, 10, "Cross-encoder", "bge-reranker-v2-m3 (GPU)\nchấm lại từng cặp (câu hỏi, tài liệu)", edge=TIER2_EDGE)
ax.text(82, 20.5, "Luôn chạy ở\nservice layer,\nkhông trong DB", ha="left", va="center",
        fontsize=8.6, color=TEXT_2, linespacing=1.35)

# --- Back to the frontend -------------------------------------------------------------------------
# Routed outside every region so it crosses nothing.
RET = -3.5
ax.plot([20, RET, RET], [20.5, 20.5, 108], color=TIER2_EDGE, linewidth=1.6)
ax.add_patch(FancyArrowPatch((RET, 108), (20, 108), arrowstyle="-|>", mutation_scale=13,
                             color=TIER2_EDGE, linewidth=1.6, shrinkA=0, shrinkB=0))
ax.text(RET - 2.4, 62, "top-10 kết quả + thời gian từng tầng", rotation=90, ha="center", va="center",
        fontsize=8.8, color=TEXT_2, style="italic")

ax.set_title("Kiến trúc hệ thống tìm kiếm hai tầng (mode hybrid_rerank)", loc="left",
             fontsize=12, fontweight="bold", color=TEXT, pad=6)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print(f"Saved {OUT}")
