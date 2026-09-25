import { useEffect, useState } from "react";
import {
  answerSpan,
  datasetQueries,
  errorMessage,
  MODE_LABEL,
  MODES,
  search,
  type Mode,
  type QueryItem,
  type SearchResponse,
} from "../api";
import CompareTable, { type CompareRow } from "./CompareTable";
import { EmptyState, SkeletonList } from "./common";
import { CompareIcon, SearchIcon, SparkIcon } from "./icons";
import ResultList from "./ResultList";
import styles from "./SearchTab.module.scss";

export interface SearchRequest {
  text: string;
  nonce: number;
}

const TOP_K = [5, 10, 20];
const MODE_HINT: Record<Mode, string> = {
  bm25: "Từ khoá",
  dense: "Ngữ nghĩa",
  hybrid: "BM25 + Dense",
  hybrid_rerank: "Cross-encoder",
};
const N_SUGGESTIONS = 3;

type View =
  | { kind: "idle" }
  | { kind: "loading"; compare: boolean }
  | { kind: "error"; message: string }
  | { kind: "results"; data: SearchResponse; span: string | null }
  | { kind: "compare"; query: string; rows: CompareRow[] };

// Deep link, e.g. /?q=Gandhi%20sinh%20năm%20nào&mode=dense (add &compare=1 for the 4-mode table).
const linked = new URLSearchParams(location.search);
const linkedMode = MODES.find((m) => m === linked.get("mode"));

export default function SearchTab({ request }: { request: SearchRequest | null }) {
  const [query, setQuery] = useState(linked.get("q") ?? "");
  const [mode, setMode] = useState<Mode>(linkedMode ?? "hybrid_rerank");
  const [k, setK] = useState(10);
  const [view, setView] = useState<View>({ kind: "idle" });
  const suggestions = useSuggestions();

  async function runSearch(q: string) {
    if (!q.trim()) return;
    setView({ kind: "loading", compare: false });
    try {
      const data = await search(q.trim(), mode, k);
      setView({ kind: "results", data, span: await answerSpan(data.qid) });
    } catch (err) {
      setView({ kind: "error", message: errorMessage(err) });
    }
  }

  async function runCompare() {
    const q = query.trim();
    if (!q) return;
    setView({ kind: "loading", compare: true });
    // One at a time: the backend serialises requests on a single DB connection anyway.
    const rows: CompareRow[] = [];
    for (const m of MODES) {
      try {
        rows.push({ mode: m, data: await search(q, m, k) });
      } catch (err) {
        rows.push({ mode: m, error: errorMessage(err) });
      }
    }
    setView({ kind: "compare", query: q, rows });
  }

  function pick(text: string) {
    setQuery(text);
    runSearch(text);
  }

  useEffect(() => {
    if (!query.trim()) return;
    if (linked.get("compare")) runCompare();
    else runSearch(query);
  }, []);

  // A question picked in the Dataset tab. Deliberately keyed on `request` only (not mode/k).
  useEffect(() => {
    if (request) pick(request.text);
  }, [request]);

  const busy = view.kind === "loading";

  return (
    <section>
      <div className={`${styles.hero} ${view.kind === "idle" ? styles.heroIdle : ""}`}>
        {view.kind === "idle" && (
          <>
            <h1 className={styles.title}>
              Hỏi bằng tiếng Việt, <span className={styles.gradient}>tìm theo ý nghĩa</span>
            </h1>
            <p className={styles.subtitle}>
              Tìm đoạn văn Wikipedia trả lời câu hỏi của bạn: retrieval lai BM25 + embedding, sắp xếp lại bằng
              cross-encoder.
            </p>
          </>
        )}

        <form
          className={styles.searchBox}
          onSubmit={(e) => {
            e.preventDefault();
            runSearch(query);
          }}
        >
          <SearchIcon size={20} className={styles.searchIcon} />
          <input
            type="search"
            aria-label="Câu hỏi"
            placeholder="Nhập câu hỏi, ví dụ: Gandhi sinh năm nào?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <button type="submit" className={styles.primary} disabled={busy || !query.trim()}>
            Tìm
          </button>
        </form>

        <div className={styles.controls}>
          <div className={styles.segmented} role="radiogroup" aria-label="Chế độ tìm kiếm">
            {MODES.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={m === mode}
                className={m === mode ? styles.selected : undefined}
                onClick={() => setMode(m)}
              >
                <span className={styles.modeName}>{MODE_LABEL[m]}</span>
                <span className={styles.modeHint}>{MODE_HINT[m]}</span>
              </button>
            ))}
          </div>
          <div className={styles.controlsRight}>
            <label className={styles.topk}>
              <span>Top</span>
              <select value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="Số kết quả">
                {TOP_K.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className={styles.ghost} onClick={runCompare} disabled={busy || !query.trim()}>
              <CompareIcon size={16} /> So sánh 4 mode
            </button>
          </div>
        </div>

        {view.kind === "idle" && suggestions.length > 0 && (
          <div className={styles.suggestions}>
            <span className={styles.suggestLabel}>
              <SparkIcon size={14} /> Thử câu hỏi trong dataset
            </span>
            {suggestions.map((s) => (
              <button key={s.qid} type="button" className={styles.chip} onClick={() => pick(s.text)}>
                {s.text}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className={styles.output}>
        {view.kind === "loading" && (
          <>
            <p className={styles.status}>{view.compare ? "Đang chạy lần lượt 4 mode..." : "Đang tìm..."}</p>
            <SkeletonList count={view.compare ? 2 : 3} />
          </>
        )}
        {view.kind === "error" && (
          <EmptyState icon={<SearchIcon size={24} />} title="Không gọi được API">
            {view.message}. Kiểm tra server <code>uvicorn rag.api:app</code> đang chạy.
          </EmptyState>
        )}
        {view.kind === "results" && <ResultList data={view.data} span={view.span} />}
        {view.kind === "compare" && <CompareTable query={view.query} rows={view.rows} k={k} />}
      </div>
    </section>
  );
}

/** A few random dataset questions for the idle screen (they come with ground truth, so hits get a ✓). */
function useSuggestions(): QueryItem[] {
  const [items, setItems] = useState<QueryItem[]>([]);
  useEffect(() => {
    datasetQueries("", 0, 1)
      .then(({ total }) =>
        Promise.all(
          Array.from({ length: N_SUGGESTIONS }, () =>
            datasetQueries("", Math.floor(Math.random() * total), 1).then((p) => p.items[0]),
          ),
        ),
      )
      .then((picked) => setItems(picked.filter((it, i) => it && picked.findIndex((p) => p?.qid === it.qid) === i)))
      .catch(() => setItems([])); // suggestions are optional
  }, []);
  return items;
}
