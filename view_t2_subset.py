"""Browse Vietnamese T2Ranking and VieQuADRetrieval queries locally.

Run: python view_t2_subset.py
Then open http://127.0.0.1:8765/ if the browser does not open automatically.
"""

import argparse
import csv
import json
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import duckdb


DATA_ROOT = Path(__file__).parent / "data"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return {row["_id"]: row for row in map(json.loads, file)}


def load_t2_data():
    root = DATA_ROOT / "t2ranking_vi" / "subset_150"
    queries = read_jsonl(root / "queries.jsonl")
    corpus = read_jsonl(root / "corpus.jsonl")
    by_query = defaultdict(list)
    with (root / "qrels.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            qid, pid = row["query-id"], row["corpus-id"]
            by_query[qid].append(
                {"id": pid, "score": int(row["score"]), "text": corpus[pid]["text"]}
            )
    return queries, by_query


def load_viequad_data():
    root = DATA_ROOT / "VieQuADRetrieval"

    def parquet(kind):
        return str(next((root / kind).glob("*.parquet")))

    queries = {
        qid: {"text": text}
        for qid, text in duckdb.sql("SELECT _id, text FROM read_parquet(?)", params=[parquet("queries")]).fetchall()
    }
    corpus = {
        pid: {"text": text, "title": title}
        for pid, text, title in duckdb.sql(
            "SELECT _id, text, title FROM read_parquet(?)", params=[parquet("corpus")]
        ).fetchall()
    }
    by_query = defaultdict(list)
    for qid, pid, score in duckdb.sql(
        'SELECT "query-id", "corpus-id", score FROM read_parquet(?)',
        params=[parquet("qrels")],
    ).fetchall():
        by_query[qid].append(
            {"id": pid, "score": int(score), "title": corpus[pid]["title"], "text": corpus[pid]["text"]}
        )
    return queries, by_query


def make_dataset(queries, by_query, label, relevant_scores, grades, grade_labels, note):
    for passages in by_query.values():
        passages.sort(key=lambda item: (-item["score"], int(item["id"])))
    return {
        "queries": queries,
        "by_query": by_query,
        "query_list": [
            {"id": qid, "text": row["text"]}
            for qid, row in sorted(queries.items(), key=lambda pair: int(pair[0]))
        ],
        "meta": {
            "label": label,
            "query_count": len(queries),
            "relevant_scores": relevant_scores,
            "grades": grades,
            "grade_labels": grade_labels,
            "note": note,
        },
    }


DATASETS = {
    "t2": make_dataset(
        *load_t2_data(),
        label="T²Ranking subset",
        relevant_scores=[2, 3],
        grades=[3, 2, 1, 0],
        grade_labels={3: "Đáp ứng rõ nhất", 2: "Liên quan một phần", 1: "Liên quan chủ đề", 0: "Không đáp ứng"},
        note="Nhãn 2–3 được tính là liên quan. Nội dung được dịch máy và nhãn kế thừa từ dữ liệu gốc.",
    ),
    "viequad": make_dataset(
        *load_viequad_data(),
        label="VieQuADRetrieval",
        relevant_scores=[1],
        grades=[1],
        grade_labels={1: "Tài liệu liên quan"},
        note="Nhãn 1 là liên quan. Mỗi query có hai tài liệu được gắn nhãn: đoạn ngữ cảnh và câu trả lời ngắn.",
    ),
}

PAGE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Xem nhãn tài liệu retrieval</title>
<style>
  :root { color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f5f6f8; color: #17212f; }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; }
  header { background: #14263e; color: #fff; padding: 26px max(24px, calc((100vw - 1060px)/2)); }
  header h1 { margin: 0 0 7px; font-size: 1.7rem; }
  header p { margin: 0; color: #c9d6e5; }
  main { max-width: 1060px; margin: 26px auto 60px; padding: 0 22px; }
  .panel { background: #fff; border: 1px solid #dfe5ec; border-radius: 14px; padding: 20px; box-shadow: 0 3px 12px #21364a0a; }
  .controls { display: grid; grid-template-columns: minmax(180px, 1fr) minmax(220px, 1fr) minmax(300px, 2fr); gap: 14px; }
  label { display: block; font-weight: 650; margin-bottom: 7px; }
  input[type=search], select { width: 100%; padding: 11px 12px; font: inherit; border: 1px solid #b9c5d2; border-radius: 8px; background: #fff; color: #17212f; }
  input:focus, select:focus { outline: 2px solid #3c7ac8; outline-offset: 1px; }
  .toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 10px 18px; margin-top: 15px; color: #3b4d61; }
  .toolbar label { display: inline; margin: 0; font-weight: 500; }
  .toolbar input { vertical-align: middle; }
  #query-title { margin: 24px 0 14px; font-size: 1.35rem; line-height: 1.4; }
  .summary { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 7px 10px; border-radius: 999px; background: #fff; border: 1px solid #dce3ea; font-size: .92rem; }
  .chip strong { font-variant-numeric: tabular-nums; }
  .groups { display: grid; gap: 20px; }
  .group h3 { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; font-size: 1.06rem; }
  .badge { border-radius: 999px; padding: 3px 9px; font-size: .8rem; font-weight: 700; }
  .s3 { background: #ddf5e7; color: #096839; }
  .s2 { background: #e4f0ff; color: #12579a; }
  .s1 { background: #fff2d8; color: #855700; }
  .s0 { background: #eeeef1; color: #566071; }
  details { background: #fff; border: 1px solid #dfe5ec; border-radius: 10px; margin-bottom: 9px; overflow: hidden; }
  summary { cursor: pointer; padding: 14px 16px; line-height: 1.5; }
  summary:hover { background: #f8fafc; }
  .pid { font-weight: 700; color: #52647b; margin-right: 8px; white-space: nowrap; }
  .full-text { border-top: 1px solid #edf0f4; padding: 14px 16px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
  .note { color: #596a7e; font-size: .91rem; margin: 18px 0 0; }
  @media (max-width: 680px) { .controls { grid-template-columns: 1fr; } header { padding: 24px 22px; } }
</style>
</head>
<body>
<header><h1>Xem nhãn tài liệu retrieval</h1><p id="dataset-description">Đang tải dataset…</p></header>
<main>
  <section class="panel">
    <div class="controls">
      <div><label for="dataset-select">Dataset</label><select id="dataset-select"></select></div>
      <div><label for="search">Tìm query theo ID hoặc nội dung</label><input id="search" type="search" placeholder="Ví dụ: Hiroshima"></div>
      <div><label for="query-select">Chọn query</label><select id="query-select"></select></div>
    </div>
    <div class="toolbar">
      <label id="show-all-control"><input id="show-all" type="checkbox"> Hiện cả nhãn 0 và 1</label>
      <span id="filter-count"></span>
    </div>
  </section>
  <h2 id="query-title">Đang tải…</h2>
  <div id="summary" class="summary"></div>
  <div id="groups" class="groups"></div>
  <p id="dataset-note" class="note"></p>
</main>
<script>
const search = document.getElementById('search');
const select = document.getElementById('query-select');
const datasetSelect = document.getElementById('dataset-select');
const showAll = document.getElementById('show-all');
const groups = document.getElementById('groups');
const title = document.getElementById('query-title');
const summary = document.getElementById('summary');
let queries = [];
let current = null;
let datasets = {};
let requestNumber = 0;

function dataset() { return datasets[datasetSelect.value]; }

function el(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function refreshOptions() {
  const needle = search.value.trim().toLocaleLowerCase('vi');
  const selected = select.value;
  const matches = queries.filter(q => (q.id + ' ' + q.text).toLocaleLowerCase('vi').includes(needle));
  select.replaceChildren();
  for (const q of matches) {
    const option = el('option', '', `${q.id} · ${q.text}`);
    option.value = q.id;
    select.append(option);
  }
  if (matches.some(q => q.id === selected)) select.value = selected;
  document.getElementById('filter-count').textContent = `${matches.length} query khớp`;
}

function render() {
  if (!current) return;
  title.textContent = `Query ${current.id}: ${current.text}`;
  const meta = dataset();
  const counts = Object.fromEntries(meta.grades.map(score => [score, 0]));
  current.docs.forEach(d => counts[d.score]++);
  summary.replaceChildren();
  const totalRelevant = meta.relevant_scores.reduce((sum, score) => sum + counts[score], 0);
  const summaryItems = [['Tài liệu liên quan', totalRelevant], ...meta.grades.map(score => [`Nhãn ${score}`, counts[score]])];
  for (const [name, value] of summaryItems) {
    const chip = el('span', 'chip');
    chip.append(el('span', '', name), el('strong', '', String(value)));
    summary.append(chip);
  }
  groups.replaceChildren();
  const visibleGrades = showAll.checked ? meta.grades : meta.grades.filter(score => meta.relevant_scores.includes(score));
  for (const score of visibleGrades) {
    const docs = current.docs.filter(d => d.score === score);
    const section = el('section', 'group');
    const heading = el('h3');
    heading.append(el('span', `badge s${score}`, `Nhãn ${score}`), el('span', '', `${meta.grade_labels[score]} · ${docs.length} tài liệu`));
    section.append(heading);
    for (const doc of docs) {
      const card = el('details');
      const line = el('summary');
      const preview = (doc.title ? `${doc.title} · ` : '') + doc.text;
      line.append(el('span', 'pid', `#${doc.id}`), document.createTextNode(preview.slice(0, 210) + (preview.length > 210 ? '…' : '')));
      const full = el('div', 'full-text');
      if (doc.title) full.append(el('strong', '', `${doc.title}\n`));
      full.append(document.createTextNode(doc.text));
      card.append(line, full);
      section.append(card);
    }
    groups.append(section);
  }
}

async function loadQuery(id) {
  if (!id) return;
  const ticket = ++requestNumber;
  const datasetId = datasetSelect.value;
  const response = await fetch(`/api/query/${encodeURIComponent(id)}?dataset=${encodeURIComponent(datasetId)}`);
  if (!response.ok) throw new Error(`Không tải được query ${id}`);
  const result = await response.json();
  if (ticket !== requestNumber || datasetSelect.value !== datasetId) return;
  current = result;
  const url = new URL(location.href);
  url.searchParams.set('dataset', datasetId);
  url.searchParams.set('q', id);
  history.replaceState(null, '', url);
  render();
}

search.addEventListener('input', refreshOptions);
select.addEventListener('change', () => loadQuery(select.value).catch(e => { title.textContent = e.message; }));
showAll.addEventListener('change', render);
datasetSelect.addEventListener('change', () => loadDataset().catch(e => { title.textContent = e.message; }));

async function loadDataset(preferredQuery) {
  ++requestNumber;
  const datasetId = datasetSelect.value;
  current = null;
  title.textContent = 'Đang tải query…';
  summary.replaceChildren();
  groups.replaceChildren();
  search.value = '';
  const meta = dataset();
  document.getElementById('dataset-description').textContent = `${meta.label} · ${meta.query_count} query tiếng Việt · Nhãn ${meta.grades.join(', ')}`;
  document.getElementById('dataset-note').textContent = meta.note;
  document.getElementById('show-all-control').hidden = meta.grades.every(score => meta.relevant_scores.includes(score));
  showAll.checked = false;
  const response = await fetch(`/api/queries?dataset=${encodeURIComponent(datasetId)}`);
  if (!response.ok) throw new Error(`Không tải được dataset ${datasetId}`);
  const result = await response.json();
  if (datasetSelect.value !== datasetId) return;
  queries = result;
  refreshOptions();
  const id = queries.some(q => q.id === preferredQuery) ? preferredQuery : queries[0]?.id;
  select.value = id;
  await loadQuery(id);
}

async function init() {
  datasets = await (await fetch('/api/datasets')).json();
  for (const [id, meta] of Object.entries(datasets)) {
    const option = el('option', '', meta.label);
    option.value = id;
    datasetSelect.append(option);
  }
  const params = new URL(location.href).searchParams;
  datasetSelect.value = datasets[params.get('dataset')] ? params.get('dataset') : 't2';
  await loadDataset(params.get('q'));
}
init().catch(e => { title.textContent = `Lỗi: ${e.message}`; });
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def send_content(self, content, content_type="application/json; charset=utf-8", status=200):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        dataset_id = parse_qs(parsed.query).get("dataset", ["t2"])[0]
        if path == "/":
            self.send_content(PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/datasets":
            self.send_content(json.dumps({key: data["meta"] for key, data in DATASETS.items()}, ensure_ascii=False))
            return
        if dataset_id not in DATASETS:
            self.send_content('{"error":"Unknown dataset"}', status=400)
            return
        data = DATASETS[dataset_id]
        if path == "/api/queries":
            self.send_content(json.dumps(data["query_list"], ensure_ascii=False))
        elif path.startswith("/api/query/"):
            qid = unquote(path.removeprefix("/api/query/"))
            if qid not in data["queries"]:
                self.send_content('{"error":"Unknown query ID"}', status=404)
                return
            payload = {"id": qid, "text": data["queries"][qid]["text"], "docs": data["by_query"][qid]}
            self.send_content(json.dumps(payload, ensure_ascii=False))
        else:
            self.send_content('{"error":"Not found"}', status=404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Mở {url} — nhấn Ctrl+C để dừng", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
