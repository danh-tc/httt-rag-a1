# Vietnamese T²Ranking subset for A1

This subset is drawn from the [Vietnamese translated THUIR T²Ranking dataset](https://huggingface.co/datasets/5CD-AI/Vietnamese-THUIR-T2Ranking-gg-translated), using its **dev** queries and graded relevance judgments. The [original T²Ranking paper](https://arxiv.org/abs/2304.03679) defines grades 0–3. It is a local teaching subset, so its scores are **not comparable** with results on the full benchmark.

| File | Contents |
| --- | --- |
| `queries.jsonl` | 150 queries, each with `_id` and Vietnamese `text` |
| `corpus.jsonl` | 2,984 Vietnamese passages, each with `_id`, empty `title`, and `text` |
| `qrels.tsv` | 2,384 query–passage judgments, columns `query-id`, `corpus-id`, `score` |
| `manifest.json` | Sampling seed, counts, and exclusions |

The query IDs were sampled with seed 42 from dev queries that originally had all four grades. All their judgments were retained where the translated passage text exists. Three judgments were excluded because the corresponding Vietnamese text is empty. Every selected query still has examples of grades 0, 1, 2, and 3. The corpus also contains 603 random extra passages from the dataset viewer as distractors.

This is machine-translated text; some Vietnamese passages and inherited relevance labels may sound awkward or disagree on individual examples. Inspect a few cases before using them in an error analysis.

For binary Recall@k or MRR, count **grades 2 and 3** as relevant. For nDCG@k, use the original 0–3 grades. The extra distractors have no judgments for these queries. As with any pooled qrels, an unjudged document may be relevant; mention this limit when interpreting scores. Avoid tuning retrieval settings on the same 150 queries used for final evaluation.

From the workspace root:

```powershell
python view_t2_subset.py
```

This opens a local interface at `http://127.0.0.1:8765/`. The **Dataset** selector switches between this T²Ranking subset and `data/VieQuADRetrieval`. Search by query ID or text, choose a query, and expand the passages to read them. For T²Ranking, grades 2–3 appear by default; check **Hiện cả nhãn 0 và 1** to see all judged passages. For VieQuADRetrieval, its score 1 means relevant, and each query has two labeled documents. The viewer requires `duckdb` (`pip install duckdb`) to read the VieQuADRetrieval Parquet files. Press Ctrl+C in the terminal to stop the server.

For a terminal-only view:

```powershell
python inspect_t2_subset.py --list
python inspect_t2_subset.py 1214 --max-per-grade 2
```

To regenerate, keep the three TSV source files in `data/t2ranking_vi/source/`, run `python prepare_t2_subset.py`, then `python fetch_t2_corpus.py`. The second command requires `requests` and network access to the Hugging Face Dataset Viewer API. The source TSVs and resulting JSONL files use UTF-8.
