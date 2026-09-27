"""Figures for the evaluation section of the report, drawn from results/eval_full.json.

    python -m rag.eval_figures          # -> results/figures/ndcg_labels.png
"""

import json

from rag.dataset_stats import COLORS, EVAL_PATH, FIGURES_DIR, GRID, PARAGRAPH, SURFACE, TEXT, TEXT_2, _style

MODE_LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + Rerank"}
ORIGINAL = "#898781"  # neutral: the original labels mix paragraph and span, so neither entity colour fits


def plot_ndcg_labels(results: dict, out_dir=FIGURES_DIR):
    """nDCG@10 per mode under both label sets; shades the Dense/Hybrid pair whose order flips."""
    plt = _style()
    modes = list(MODE_LABELS)
    series = [
        ("Nhãn gốc (đoạn văn + cụm câu trả lời)", "ndcg", ORIGINAL),
        ("Chỉ đoạn văn", "ndcg_para", COLORS[PARAGRAPH]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(0.5, 2.5, color=GRID, alpha=0.45, linewidth=0, zorder=0)  # Dense + Hybrid groups
    width = 0.38
    for offset, (label, key, color) in zip((-width / 2, width / 2), series):
        values = [results[m]["agg"][key] for m in modes]
        bars = ax.bar([i + offset for i in range(len(modes))], values, width=width, label=label,
                      color=color, edgecolor=SURFACE, linewidth=1.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=8.5, color=TEXT_2, padding=2)

    d, h = (results[m]["agg"] for m in ("dense", "hybrid"))
    ax.text(1.5, 1.2, "Thứ tự đảo chiều", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=TEXT)
    ax.text(1.5, 1.18,
            f"Nhãn gốc: Dense {d['ndcg']:.3f} > Hybrid {h['ndcg']:.3f}\n"
            f"Chỉ đoạn văn: Hybrid {h['ndcg_para']:.3f} > Dense {d['ndcg_para']:.3f}",
            ha="center", va="top", fontsize=8.5, color=TEXT_2, linespacing=1.4)

    ax.set_xticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
    ax.set_ylim(0, 1.27)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("nDCG@10")
    ax.set_title("nDCG@10 theo hai bộ nhãn")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)  # below: the top holds the note
    fig.tight_layout()
    path = out_dir / "ndcg_labels.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    results = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"Saved {plot_ndcg_labels(results)}")


if __name__ == "__main__":
    main()
