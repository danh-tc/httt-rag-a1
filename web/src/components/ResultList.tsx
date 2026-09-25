import { MODE_LABEL, type SearchResponse } from "../api";
import { DeviceBadge, DocCard, EmptyState, Highlight, HitBadge, KindTag } from "./common";
import { CheckIcon, ClockIcon, SearchIcon } from "./icons";
import styles from "./ResultList.module.scss";
import Score from "./Score";

/** Top-k results of one search; `span` is the answer text to highlight in the judged paragraph. */
export default function ResultList({ data, span }: { data: SearchResponse; span: string | null }) {
  const judged = data.judged ?? {};
  const nJudged = Object.keys(judged).length;
  const hits = data.results.filter((r) => r.doc_id in judged).length;
  const rerankShare = data.took_ms > 0 ? (data.rerank_ms / data.took_ms) * 100 : 0;

  return (
    <>
      <div className={styles.summary}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Mode</span>
          <span className={styles.statValue}>{MODE_LABEL[data.mode]}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>
            <ClockIcon size={12} /> Thời gian
          </span>
          <span className={styles.statValue}>{data.took_ms.toFixed(0)} ms</span>
          <div className={styles.timeBar} title={`retrieval ${data.retrieval_ms.toFixed(0)} ms + rerank ${data.rerank_ms.toFixed(0)} ms`}>
            <span className={styles.retrieval} style={{ width: `${100 - rerankShare}%` }} />
            <span className={styles.rerank} style={{ width: `${rerankShare}%` }} />
          </div>
        </div>
        {data.qid && (
          <div className={styles.stat}>
            <span className={styles.statLabel}>
              <CheckIcon size={12} /> Câu hỏi #{data.qid}
            </span>
            <span className={styles.statValue}>
              {hits}/{nJudged} <small>tài liệu đúng trong top-{data.results.length}</small>
            </span>
          </div>
        )}
        <div className={styles.device}>
          <DeviceBadge device={data.device} />
        </div>
      </div>

      {data.results.length === 0 ? (
        <EmptyState icon={<SearchIcon size={24} />} title="Không tìm thấy kết quả">
          Thử diễn đạt lại câu hỏi hoặc đổi sang mode Dense.
        </EmptyState>
      ) : (
        data.results.map((r, i) => {
          const hit = r.doc_id in judged;
          return (
            <DocCard
              key={r.doc_id}
              rank={i + 1}
              kind={r.kind}
              hit={hit}
              delay={i * 30}
              title={r.title || `Tài liệu #${r.doc_id}`}
              aside={<Score mode={data.mode} value={r.score} />}
              tags={
                <>
                  <KindTag kind={r.kind} />
                  {hit && <HitBadge />}
                  <span className={styles.docId}>#{r.doc_id}</span>
                </>
              }
              body={hit && r.kind === "paragraph" ? <Highlight text={r.body} needle={span} /> : r.body}
            />
          );
        })
      )}
    </>
  );
}
