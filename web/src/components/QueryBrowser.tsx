import { useEffect, useState } from "react";
import { datasetQueries, datasetQuery, errorMessage, KIND_LABEL, type QueryDetail, type QueryPage } from "../api";
import { DocCard, EmptyState, Highlight, KindTag } from "./common";
import { ArrowRightIcon, ChevronIcon, SearchIcon } from "./icons";
import styles from "./QueryBrowser.module.scss";

const PAGE_SIZE = 20;

/** Filterable, paged list of dataset questions; picking one shows its ground-truth documents. */
export default function QueryBrowser({ onSearch }: { onSearch: (text: string) => void }) {
  const [filter, setFilter] = useState("");
  const [debounced, setDebounced] = useState("");
  const [page, setPage] = useState(0);
  const [data, setData] = useState<QueryPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeQid, setActiveQid] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebounced(filter.trim());
      setPage(0);
    }, 250);
    return () => clearTimeout(timer);
  }, [filter]);

  useEffect(() => {
    let stale = false;
    datasetQueries(debounced, page * PAGE_SIZE, PAGE_SIZE).then(
      (d) => !stale && (setData(d), setError(null)),
      (err) => !stale && setError(errorMessage(err)),
    );
    return () => {
      stale = true;
    };
  }, [debounced, page]);

  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE));

  return (
    <div className={styles.browser}>
      <div className={styles.listPane}>
        <div className={styles.filter}>
          <SearchIcon size={16} />
          <input
            type="search"
            aria-label="Lọc câu hỏi"
            placeholder="Lọc theo nội dung hoặc ID..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <div className={styles.list}>
          {error && <p className={styles.empty}>Lỗi: {error}</p>}
          {data?.items.length === 0 && <p className={styles.empty}>Không có câu hỏi nào khớp.</p>}
          {data?.items.map((item) => (
            <button
              key={item.qid}
              type="button"
              className={item.qid === activeQid ? styles.active : undefined}
              onClick={() => setActiveQid(item.qid)}
            >
              <span className={styles.qid}>#{item.qid}</span>
              <span className={styles.qtext}>{item.text}</span>
            </button>
          ))}
        </div>
        <div className={styles.pager}>
          <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Trang trước">
            <ChevronIcon dir="left" size={16} />
          </button>
          <span>
            Trang {page + 1}/{pages} · {(data?.total ?? 0).toLocaleString("vi-VN")} câu hỏi
          </span>
          <button type="button" disabled={page >= pages - 1} onClick={() => setPage(page + 1)} aria-label="Trang sau">
            <ChevronIcon size={16} />
          </button>
        </div>
      </div>

      <div className={styles.detailPane}>
        {activeQid ? (
          <QueryDetailView qid={activeQid} onSearch={onSearch} />
        ) : (
          <EmptyState icon={<SearchIcon size={24} />} title="Chọn một câu hỏi">
            để xem đoạn văn và cụm câu trả lời được gán nhãn đúng.
          </EmptyState>
        )}
      </div>
    </div>
  );
}

function QueryDetailView({ qid, onSearch }: { qid: string; onSearch: (text: string) => void }) {
  const [detail, setDetail] = useState<QueryDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    setDetail(null);
    setError(null);
    datasetQuery(qid).then(
      (d) => !stale && setDetail(d),
      (err) => !stale && setError(errorMessage(err)),
    );
    return () => {
      stale = true;
    };
  }, [qid]);

  if (error) return <p className={styles.empty}>Lỗi: {error}</p>;
  if (!detail) return <p className={styles.empty}>Đang tải...</p>;
  const span = detail.relevant.find((d) => d.kind === "span")?.body;

  return (
    <div>
      <div className={styles.eyebrow}>Câu hỏi #{detail.qid}</div>
      <div className={styles.question}>{detail.text}</div>
      <button type="button" className={styles.primary} onClick={() => onSearch(detail.text)}>
        Tìm với câu hỏi này <ArrowRightIcon size={16} />
      </button>
      <div className={styles.subheading}>Tài liệu đúng</div>
      {detail.relevant.map((doc) => (
        <DocCard
          key={doc.doc_id}
          kind={doc.kind}
          title={doc.title || KIND_LABEL[doc.kind]}
          tags={
            <>
              <KindTag kind={doc.kind} />
              <span className={styles.docId}>#{doc.doc_id}</span>
            </>
          }
          body={doc.kind === "paragraph" ? <Highlight text={doc.body} needle={span} /> : doc.body}
        />
      ))}
      <p className={styles.note}>
        Phần <mark>tô vàng</mark> trong đoạn văn chính là cụm câu trả lời.
      </p>
    </div>
  );
}
