"""Profile VieQuADRetrieval for the report: summary table, JSON, and PNG figures.

    python -m rag.dataset_stats          # -> results/dataset_stats.json + results/figures/*.png

Key property of this dataset: each query is judged relevant to exactly two documents,
the *context paragraph* and the *answer span* (a short phrase cut out of that paragraph).
Spans share few words with the question, which caps what lexical retrieval can reach.
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from rag.config import RESULTS_DIR
from rag.dataset import load_corpus, load_qrels, load_queries

FIGURES_DIR = RESULTS_DIR / "figures"
STATS_PATH = RESULTS_DIR / "dataset_stats.json"
EVAL_PATH = RESULTS_DIR / "eval_full.json"

PARAGRAPH, SPAN = "paragraph", "span"
KIND_LABELS = {PARAGRAPH: "Đoạn văn (ngữ cảnh)", SPAN: "Cụm câu trả lời"}


def tokens(text: str) -> list[str]:
    # Vietnamese is written syllable-by-syllable, so this counts syllables, not words.
    return re.findall(r"\w+", text.lower())


def doc_kinds(corpus: dict, qrels: dict) -> dict[str, str]:
    """Label each judged doc: the one contained in its query partner is the answer span."""
    kinds: dict[str, str] = {}
    for relevance in qrels.values():
        doc_a, doc_b = relevance
        text_a, text_b = corpus[doc_a]["text"], corpus[doc_b]["text"]
        if text_a == text_b:
            continue
        if text_a in text_b:
            span, para = doc_a, doc_b
        elif text_b in text_a:
            span, para = doc_b, doc_a
        else:
            continue
        kinds[span] = SPAN
        kinds.setdefault(para, PARAGRAPH)
    return kinds


def overlap(query: str, doc: str) -> float:
    """Share of the query's distinct syllables that also occur in the doc."""
    q = set(tokens(query))
    return len(q & set(tokens(doc))) / len(q) if q else 0.0


def _describe(values: list[float]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 2),
        "max": max(values),
    }


def profile(corpus: dict, queries: dict, qrels: dict) -> dict:
    """Everything the report and the frontend need, as plain JSON-able data."""
    kinds = doc_kinds(corpus, qrels)
    lengths = {kind: [len(tokens(corpus[d]["text"])) for d, k in kinds.items() if k == kind] for kind in KIND_LABELS}
    overlaps = {kind: [] for kind in KIND_LABELS}
    for qid, relevance in qrels.items():
        for doc_id in relevance:
            overlaps[kinds[doc_id]].append(overlap(queries[qid], corpus[doc_id]["text"]))

    per_article: dict[str, Counter] = {}
    for doc_id, doc in corpus.items():
        per_article.setdefault(doc["title"], Counter())[kinds.get(doc_id, "unjudged")] += 1

    return {
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "n_judged_queries": len(qrels),
        "n_articles": len(per_article),
        "n_by_kind": dict(Counter(kinds.values())),
        "relevant_per_query": dict(Counter(len(r) for r in qrels.values())),
        "queries_per_paragraph": _describe(
            list(Counter(d for r in qrels.values() for d in r if kinds[d] == PARAGRAPH).values())
        ),
        "doc_length": {kind: _describe(v) for kind, v in lengths.items()},
        "query_length": _describe([len(tokens(q)) for q in queries.values()]),
        "overlap": {kind: _describe(v) for kind, v in overlaps.items()},
        "per_article": dict(
            sorted(((t, dict(c)) for t, c in per_article.items()), key=lambda item: -sum(item[1].values()))
        ),
        # Raw series for the figures; dropped from the JSON.
        "_lengths": lengths,
        "_query_lengths": [len(tokens(q)) for q in queries.values()],
        "_overlaps": overlaps,
        "_kinds": kinds,
    }


def found_by_kind(eval_results: dict, kinds: dict[str, str]) -> dict[str, dict[str, float]]:
    """Per mode: share of queries whose paragraph / answer span made the top-k."""
    out = {}
    for mode, result in eval_results.items():
        per_query = result["per_query"].values()
        out[mode] = {
            kind: sum(any(kinds[d] == kind and d in q["retrieved"] for d in q["relevant"]) for q in per_query)
            / len(per_query)
            for kind in KIND_LABELS
        }
    return out


# --- figures -------------------------------------------------------------------------------------
# Reference palette (dataviz skill), validated all-pairs for 3 slots. Colour follows the entity:
# paragraph is always blue, span always orange, queries aqua.
COLORS = {PARAGRAPH: "#2a78d6", SPAN: "#eb6834", "query": "#1baf7a"}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,  # DejaVu covers Vietnamese diacritics
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
    return plt


def _bar_kw(kind: str) -> dict:
    # White edge = the 2px surface gap between adjacent fills.
    return {"color": COLORS[kind], "edgecolor": SURFACE, "linewidth": 1.5}


def plot_figures(stats: dict, found: dict | None, out_dir: Path) -> list[Path]:
    plt = _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)

    # 1. Document length, one panel per kind (their scales differ by ~10x).
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, kind, bins in zip(axes, KIND_LABELS, (range(100, 420, 20), range(0, 42, 2))):
        values = [min(v, bins[-1] - 1) for v in stats["_lengths"][kind]]
        ax.hist(values, bins=list(bins), **_bar_kw(kind))
        med = stats["doc_length"][kind]["median"]
        ax.set_title(f"{KIND_LABELS[kind]}  (n={len(values)}, trung vị {med:g} âm tiết)")
        ax.set_xlabel("Số âm tiết" + (" (gộp ≥400)" if kind == PARAGRAPH else " (gộp ≥40)"))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Số tài liệu")
    save(fig, "doc_length.png")

    # 2. Query length.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(stats["_query_lengths"], bins=range(0, 52, 2), **_bar_kw("query"))
    ax.set_title(f"Độ dài câu hỏi  (n={stats['n_queries']}, trung vị {stats['query_length']['median']:g} âm tiết)")
    ax.set_xlabel("Số âm tiết")
    ax.set_ylabel("Số câu hỏi")
    ax.grid(axis="x", visible=False)
    save(fig, "query_length.png")

    # 3. Documents per Wikipedia article, stacked by kind.
    articles = list(stats["per_article"].items())[::-1]  # largest on top
    fig, ax = plt.subplots(figsize=(8, 5.6))
    names = [t for t, _ in articles]
    left = [0] * len(articles)
    for kind in KIND_LABELS:
        counts = [c.get(kind, 0) for _, c in articles]
        ax.barh(names, counts, left=left, label=KIND_LABELS[kind], height=0.7, **_bar_kw(kind))
        left = [a + b for a, b in zip(left, counts)]
    for y, total in enumerate(left):
        ax.text(total + 3, y, str(total), va="center", fontsize=8.5, color=TEXT_2)
    ax.set_title(f"Số tài liệu theo bài Wikipedia  ({stats['n_articles']} bài, {stats['n_docs']} tài liệu)")
    ax.set_xlabel("Số tài liệu")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "docs_per_article.png")

    # 4. Lexical overlap between query and each relevant doc, grouped by kind.
    bins = [i / 10 for i in range(11)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.045
    for offset, kind in zip((-width / 2, width / 2), KIND_LABELS):
        values = stats["_overlaps"][kind]
        counts = [0] * 10
        for v in values:
            counts[min(int(v * 10), 9)] += 1
        shares = [c / len(values) * 100 for c in counts]
        ax.bar([b + 0.05 + offset for b in bins[:-1]], shares, width=width, label=KIND_LABELS[kind], **_bar_kw(kind))
    ax.set_xticks(bins, [f"{b:.0%}" for b in bins])
    ax.set_title("Tỷ lệ âm tiết của câu hỏi xuất hiện trong tài liệu đúng")
    ax.set_xlabel("Độ trùng từ vựng câu hỏi – tài liệu")
    ax.set_ylabel("% cặp (câu hỏi, tài liệu)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center")
    save(fig, "lexical_overlap.png")

    # 5. Which half of the judgment each mode actually finds (needs a full eval run).
    if found:
        modes = list(found)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.36
        for offset, kind in zip((-width / 2, width / 2), KIND_LABELS):
            shares = [found[m][kind] * 100 for m in modes]
            bars = ax.bar([i + offset for i in range(len(modes))], shares, width=width,
                          label=KIND_LABELS[kind], **_bar_kw(kind))
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8.5, color=TEXT_2, padding=2)
        ax.set_xticks(range(len(modes)), modes)
        ax.set_ylim(0, 118)  # headroom so the legend never sits on a bar
        ax.set_title("Tỷ lệ câu hỏi tìm thấy từng loại tài liệu đúng trong top-10")
        ax.set_ylabel("% câu hỏi")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=2)
        save(fig, "found_by_kind.png")

    return saved


def load_profile() -> dict:
    return profile(load_corpus(), load_queries(), load_qrels())


def public(stats: dict) -> dict:
    """Drop the raw series (keys starting with '_')."""
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stats = load_profile()
    eval_results = json.loads(EVAL_PATH.read_text(encoding="utf-8")) if EVAL_PATH.exists() else None
    found = found_by_kind(eval_results, stats["_kinds"]) if eval_results else None

    s = public(stats)
    print(f"Tài liệu: {s['n_docs']}  ({', '.join(f'{KIND_LABELS[k]}: {v}' for k, v in s['n_by_kind'].items())})")
    print(f"Bài Wikipedia: {s['n_articles']}   Câu hỏi: {s['n_queries']} (có nhãn: {s['n_judged_queries']})")
    print(f"Số tài liệu đúng / câu hỏi: {s['relevant_per_query']}")
    print(f"Số câu hỏi / đoạn văn: {s['queries_per_paragraph']}")
    print(f"{'':38s}{'min':>7s}{'median':>8s}{'mean':>8s}{'max':>7s}")
    rows = [(f"Độ dài {KIND_LABELS[k]}", s["doc_length"][k]) for k in KIND_LABELS]
    rows += [("Độ dài câu hỏi", s["query_length"])]
    rows += [(f"Trùng từ vựng – {KIND_LABELS[k]}", s["overlap"][k]) for k in KIND_LABELS]
    for label, d in rows:
        print(f"{label:38s}" + "".join(f"{round(d[key], 2):>{w}}" for key, w in (("min", 7), ("median", 8), ("mean", 8), ("max", 7))))
    if found:
        print("\nTìm thấy trong top-10 (theo loại tài liệu đúng):")
        for mode, shares in found.items():
            print(f"  {mode:15s} " + "  ".join(f"{KIND_LABELS[k]}: {v:.1%}" for k, v in shares.items()))
        s["found_by_kind"] = found

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = plot_figures(stats, found, FIGURES_DIR)
    print(f"\nSaved {STATS_PATH} and {len(saved)} figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
