"""Select reproducible Vietnamese T2Ranking dev queries and their full qrels.

Run from the workspace root: python prepare_t2_subset.py
The corpus is not written here; see the generated manifest for required passage IDs.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent / "data" / "t2ranking_vi"
SOURCE = ROOT / "source"
OUT = ROOT / "subset_150"
SEED = 42
QUERY_COUNT = 150


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


queries = {
    row["qid"]: row["text_vi"].strip()
    for row in read_tsv(SOURCE / "data_queries.dev_json_vi.tsv")
    if row["text_vi"].strip()
}
qrels = defaultdict(list)
for row in read_tsv(SOURCE / "data_qrels.dev.tsv"):
    if row["qid"] in queries:
        qrels[row["qid"]].append((row["pid"], int(row["rel"])))

# Keep questions with positive evidence and examples at each relevance grade.
eligible = [
    qid
    for qid, judgments in qrels.items()
    if {grade for _, grade in judgments} == {0, 1, 2, 3}
]
if len(eligible) < QUERY_COUNT:
    raise SystemExit(f"Only {len(eligible)} queries contain all four grades")

rng = random.Random(SEED)
selected = sorted(rng.sample(eligible, QUERY_COUNT), key=int)
judgments = [(qid, pid, grade) for qid in selected for pid, grade in qrels[qid]]
required_doc_ids = sorted({pid for _, pid, _ in judgments}, key=int)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "queries.jsonl").open("w", encoding="utf-8") as f:
    for qid in selected:
        f.write(json.dumps({"_id": qid, "text": queries[qid]}, ensure_ascii=False) + "\n")
with (OUT / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["query-id", "corpus-id", "score"])
    writer.writerows(judgments)
(OUT / "required_doc_ids.txt").write_text("\n".join(required_doc_ids) + "\n", encoding="utf-8")
manifest = {
    "source": "5CD-AI/Vietnamese-THUIR-T2Ranking-gg-translated dev",
    "seed": SEED,
    "query_count": len(selected),
    "judgment_count": len(judgments),
    "unique_judged_document_count": len(required_doc_ids),
    "grade_counts": dict(sorted(Counter(grade for _, _, grade in judgments).items())),
    "binary_relevant_grades": [2, 3],
    "status": "query and qrels selected; corpus passages still required",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
