import { useEffect, useState } from "react";
import { datasetStats, errorMessage, type DatasetStats } from "../api";
import { EmptyState, KindTag } from "./common";
import { DatabaseIcon } from "./icons";
import QueryBrowser from "./QueryBrowser";
import styles from "./DatasetTab.module.scss";

// Produced by `python -m rag.dataset_stats`; any that don't exist yet are hidden.
const FIGURES = [
  ["docs_per_article.png", "Số tài liệu theo bài Wikipedia"],
  ["doc_length.png", "Độ dài tài liệu"],
  ["query_length.png", "Độ dài câu hỏi"],
  ["lexical_overlap.png", "Độ trùng từ vựng câu hỏi – tài liệu đúng"],
  ["found_by_kind.png", "Tỷ lệ tìm thấy theo loại tài liệu"],
];

/** Dataset overview. Loads on first activation, then stays mounted. */
export default function DatasetTab({ active, onSearch }: { active: boolean; onSearch: (text: string) => void }) {
  const [opened, setOpened] = useState(active);
  useEffect(() => {
    if (active) setOpened(true);
  }, [active]);
  if (!opened) return null;

  return (
    <section>
      <div className={styles.intro}>
        <div className={styles.eyebrow}>VieQuADRetrieval</div>
        <h1 className={styles.title}>Dữ liệu</h1>
        <p className={styles.lead}>
          Câu hỏi tiếng Việt trên các bài Wikipedia, có sẵn nhãn tài liệu đúng cho mọi câu hỏi (dùng để tính Recall,
          MRR, nDCG).
        </p>
      </div>

      <StatTiles />

      <div className={styles.callout}>
        <div className={styles.calloutTitle}>Đặc điểm quan trọng</div>
        Mỗi câu hỏi có đúng <b>2 tài liệu đúng</b>: <KindTag kind="paragraph" /> ngữ cảnh và{" "}
        <KindTag kind="span" /> (một cụm ngắn cắt ra từ chính đoạn văn đó). Cụm câu trả lời gần như không chứa từ nào
        của câu hỏi, nên rất khó tìm bằng từ khoá.
      </div>

      <h2 className={styles.sectionTitle}>Biểu đồ</h2>
      <Figures />

      <h2 className={styles.sectionTitle}>Duyệt câu hỏi</h2>
      <QueryBrowser onSearch={onSearch} />
    </section>
  );
}

function StatTiles() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    datasetStats().then(setStats, (err) => setError(errorMessage(err)));
  }, []);

  if (error) return <EmptyState icon={<DatabaseIcon size={24} />} title="Không tải được thống kê">{error}</EmptyState>;
  if (!stats) return <div className={styles.tiles}>{Array.from({ length: 8 }, (_, i) => <div key={i} className={`${styles.tile} ${styles.loading}`} />)}</div>;

  const tiles: { value: number; label: string; tone?: "paragraph" | "span" | "accent" }[] = [
    { value: stats.n_docs, label: "tài liệu", tone: "accent" },
    { value: stats.n_by_kind.paragraph, label: "đoạn văn ngữ cảnh", tone: "paragraph" },
    { value: stats.n_by_kind.span, label: "cụm câu trả lời", tone: "span" },
    { value: stats.n_articles, label: "bài Wikipedia" },
    { value: stats.n_queries, label: "câu hỏi (đều có nhãn)", tone: "accent" },
    { value: stats.query_length.median, label: "âm tiết / câu hỏi (trung vị)" },
    { value: stats.doc_length.paragraph.median, label: "âm tiết / đoạn văn (trung vị)", tone: "paragraph" },
    { value: stats.doc_length.span.median, label: "âm tiết / cụm trả lời (trung vị)", tone: "span" },
  ];
  return (
    <div className={styles.tiles}>
      {tiles.map(({ value, label, tone }) => (
        <div key={label} className={`${styles.tile} ${tone ? styles[tone] : ""}`}>
          <div className={styles.value}>{value.toLocaleString("vi-VN")}</div>
          <div className={styles.label}>{label}</div>
        </div>
      ))}
    </div>
  );
}

function Figures() {
  const [missing, setMissing] = useState<Set<string>>(new Set());
  const shown = FIGURES.filter(([file]) => !missing.has(file));
  if (shown.length === 0) {
    return (
      <EmptyState icon={<DatabaseIcon size={24} />} title="Chưa có biểu đồ">
        Chạy <code>python -m rag.dataset_stats</code> để tạo.
      </EmptyState>
    );
  }
  return (
    <div className={styles.figures}>
      {shown.map(([file, caption]) => (
        <a key={file} className={styles.figure} href={`/figures/${file}`} target="_blank" rel="noreferrer">
          <div className={styles.figureImg}>
            <img
              src={`/figures/${file}`}
              alt={caption}
              loading="lazy"
              onError={() => setMissing((prev) => new Set(prev).add(file))}
            />
          </div>
          <div className={styles.caption}>{caption}</div>
        </a>
      ))}
    </div>
  );
}
