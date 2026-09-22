"""Fetch a small, reproducible Vietnamese T2Ranking corpus via Dataset Viewer.

Run after prepare_t2_subset.py: python fetch_t2_corpus.py
Requires: requests. Network access to datasets-server.huggingface.co is needed.
"""

import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).parent / "data" / "t2ranking_vi" / "subset_150"
API = "https://datasets-server.huggingface.co"
DATASET = "5CD-AI/Vietnamese-THUIR-T2Ranking-gg-translated"
COMMON = {"dataset": DATASET, "config": "collection", "split": "train"}
DISTRACTOR_COUNT = 600


def get_json(session, endpoint, params):
    for attempt in range(6):
        try:
            response = session.get(API + endpoint, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))


def passage(row):
    pid = str(row["pid"])
    content = (row["text_vi"] or "").strip()
    if not content:
        return pid, None
    return pid, {"_id": pid, "title": "", "text": content}


required_ids = set((ROOT / "required_doc_ids.txt").read_text(encoding="utf-8").splitlines())
docs = {}
with requests.Session() as session:
    ordered_ids = sorted(required_ids, key=int)
    batches = [ordered_ids[start : start + 25] for start in range(0, len(ordered_ids), 25)]

    def fetch_batch(batch):
        where = " OR ".join(f'"pid"=\'{pid}\'' for pid in batch)
        result = get_json(requests, "/filter", {**COMMON, "where": where, "offset": 0, "length": 100})
        return [item["row"] for item in result["rows"]]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_batch, batch) for batch in batches]
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                pid, doc = passage(row)
                if doc is not None:
                    docs[pid] = doc
            print(f"Judged passages: {len(docs)}/{len(required_ids)}", flush=True)

    missing = required_ids - docs.keys()
    if missing:
        print(f"Judged passages missing Vietnamese text or API row: {len(missing)}; IDs: {sorted(missing, key=int)[:30]}", flush=True)
    retained_required_count = len(required_ids) - len(missing)

    size = get_json(session, "/size", {"dataset": DATASET})
    splits = size["size"]["splits"]
    indexed_count = next(
        split["num_rows"]
        for split in splits
        if split["config"] == "collection" and split["split"] == "train"
    )
    rng = random.Random(43)
    offsets = rng.sample(range(0, indexed_count - 100), 20)
    for offset in offsets:
        if len(docs) >= retained_required_count + DISTRACTOR_COUNT:
            break
        result = get_json(session, "/rows", {**COMMON, "offset": offset, "length": 100})
        for item in result["rows"]:
            pid, doc = passage(item["row"])
            if doc is None:
                continue
            docs.setdefault(pid, doc)
            if len(docs) >= retained_required_count + DISTRACTOR_COUNT:
                break

if len(docs) < retained_required_count + DISTRACTOR_COUNT:
    raise RuntimeError("Could not obtain enough distinct distractor passages")

with (ROOT / "corpus.jsonl").open("w", encoding="utf-8") as f:
    for pid in sorted(docs, key=int):
        f.write(json.dumps(docs[pid], ensure_ascii=False) + "\n")

with (ROOT / "qrels.tsv").open(encoding="utf-8", newline="") as f:
    qrels = list(csv.DictReader(f, delimiter="\t"))
missing_judged = [row for row in qrels if row["corpus-id"] not in docs]
if missing_judged:
    qrels = [row for row in qrels if row["corpus-id"] in docs]
    with (ROOT / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
        writer.writeheader()
        writer.writerows(qrels)
    print(f"Removed {len(missing_judged)} judgments without Vietnamese passage text", flush=True)
retained_ids = {row["corpus-id"] for row in qrels}
(ROOT / "required_doc_ids.txt").write_text(
    "\n".join(sorted(retained_ids, key=int)) + "\n", encoding="utf-8"
)
assert len({row["query-id"] for row in qrels}) == 150
grades_by_query = {}
for row in qrels:
    grades_by_query.setdefault(row["query-id"], set()).add(int(row["score"]))
assert all(grades == {0, 1, 2, 3} for grades in grades_by_query.values())

manifest_path = ROOT / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update(
    judgment_count=len(qrels),
    unique_judged_document_count=len(retained_ids),
    grade_counts={str(grade): sum(int(row["score"]) == grade for row in qrels) for grade in range(4)},
    corpus_document_count=len(docs),
    distractor_document_count=len(docs) - len(retained_ids),
    excluded_empty_translation_ids=sorted(missing, key=int),
    status="ready",
)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
