import { MODE_LABEL, type Mode, type SearchResponse } from "../api";
import { KindTag } from "./common";
import styles from "./CompareTable.module.scss";
import Score from "./Score";

export type CompareRow = { mode: Mode; data: SearchResponse; error?: undefined } | { mode: Mode; data?: undefined; error: string };

/** The same query through all 4 modes: latency split, top-1, and where the judged docs landed. */
export default function CompareTable({ query, rows, k }: { query: string; rows: CompareRow[]; k: number }) {
  const judgedQid = rows.find((r) => r.data?.qid)?.data?.qid;
  const maxMs = Math.max(1, ...rows.map((r) => r.data?.took_ms ?? 0));

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <div>
          <div className={styles.eyebrow}>So sánh 4 mode{judgedQid && ` · câu hỏi #${judgedQid}`}</div>
          <div className={styles.query}>“{query}”</div>
        </div>
        <div className={styles.legend}>
          <span>
            <i className={styles.retrieval} /> retrieval
          </span>
          <span>
            <i className={styles.rerank} /> rerank
          </span>
          {judgedQid && (
            <>
              <span>
                <KindTag kind="paragraph" dot /> đoạn văn
              </span>
              <span>
                <KindTag kind="span" dot /> cụm trả lời
              </span>
            </>
          )}
        </div>
      </div>

      <div className={`${styles.grid} ${judgedQid ? styles.withHits : ""}`}>
        <div className={styles.th}>Mode</div>
        <div className={styles.th}>Thời gian</div>
        <div className={styles.th}>Top-1</div>
        {judgedQid && <div className={styles.th}>Tài liệu đúng trong top-{k}</div>}

        {rows.map((row) =>
          row.data ? (
            <Row key={row.mode} data={row.data} maxMs={maxMs} showHits={Boolean(judgedQid)} />
          ) : (
            <div key={row.mode} className={styles.errorRow}>
              <b>{MODE_LABEL[row.mode]}</b>: lỗi {row.error}
            </div>
          ),
        )}
      </div>
    </div>
  );
}

function Row({ data, maxMs, showHits }: { data: SearchResponse; maxMs: number; showHits: boolean }) {
  const top = data.results[0];
  const judged = data.judged ?? {};
  const hits = data.results.map((r, i) => ({ r, rank: i + 1 })).filter(({ r }) => r.doc_id in judged);

  return (
    <>
      <div className={styles.mode}>{MODE_LABEL[data.mode]}</div>
      <div className={styles.time}>
        <div className={styles.track}>
          <span className={styles.retrieval} style={{ width: `${(data.retrieval_ms / maxMs) * 100}%` }} />
          <span className={styles.rerank} style={{ width: `${(data.rerank_ms / maxMs) * 100}%` }} />
        </div>
        <span className={styles.ms}>{data.took_ms.toFixed(0)} ms</span>
      </div>
      <div className={styles.top}>
        {top ? (
          <>
            <span className={styles.topTitle}>{top.title || `#${top.doc_id}`}</span>
            <span className={styles.topMeta}>
              <span>#{top.doc_id}</span>
              <Score mode={data.mode} value={top.score} compact />
            </span>
          </>
        ) : (
          "—"
        )}
      </div>
      {showHits && (
        <div className={styles.hits}>
          <span className={styles.hitCount}>
            {hits.length}/{Object.keys(judged).length}
          </span>
          {hits.map(({ r, rank }) => (
            <span key={r.doc_id} className={styles.hitRank}>
              <KindTag kind={judged[r.doc_id]} dot /> #{rank}
            </span>
          ))}
        </div>
      )}
    </>
  );
}
