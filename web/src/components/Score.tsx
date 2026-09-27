import { useId, type ReactNode } from "react";
import type { Mode } from "../api";
import styles from "./Score.module.scss";

// Each mode scores on its own scale, so a bare number is meaningless without saying which one.
const SCORE_INFO: Record<Mode, { label: string; range: string; digits: number; explain: ReactNode }> = {
  bm25: {
    label: "BM25",
    range: "0 → không giới hạn (thường 5–40)",
    digits: 2,
    explain: (
      <>
        Tổng điểm của các âm tiết trong câu hỏi xuất hiện trong tài liệu. Âm tiết càng hiếm (IDF cao) càng được nhiều
        điểm; tài liệu dài bị trừ nhẹ. Không có mức trần: câu hỏi dài, nhiều từ hiếm cho điểm cao hơn.
      </>
    ),
  },
  dense: {
    label: "cosine",
    range: "−1 → 1 (với e5 thường chỉ 0.7–0.9)",
    digits: 3,
    explain: (
      <>
        Độ tương đồng cosine giữa vector câu hỏi và vector tài liệu (multilingual-e5). e5 được huấn luyện với nhiệt độ
        rất thấp nên mọi điểm dồn vào một khoảng hẹp, kể cả tài liệu không liên quan: hãy nhìn thứ hạng, đừng nhìn giá
        trị tuyệt đối.
      </>
    ),
  },
  hybrid: {
    label: "RRF",
    range: "0 → 0.033",
    digits: 4,
    explain: (
      <>
        Reciprocal Rank Fusion chỉ dùng <b>thứ hạng</b>, bỏ qua điểm gốc: 1/(60 + hạng BM25) + 1/(60 + hạng Dense). Tối
        đa 2/61 ≈ 0.033 khi đứng đầu cả hai danh sách; chỉ có mặt ở một danh sách thì tối đa ≈ 0.016.
      </>
    ),
  },
  hybrid_rerank: {
    label: "rerank",
    range: "0 → 1",
    digits: 3,
    explain: (
      <>
        Xác suất liên quan do cross-encoder (bge-reranker-v2-m3) chấm khi đọc câu hỏi và tài liệu <b>cùng lúc</b>
        (logit qua hàm sigmoid). Điểm thường tách rõ: tài liệu đúng gần 1, tài liệu sai gần 0.
      </>
    ),
  },
};

/** A score labelled with its scale, with a tooltip explaining what the number means for `mode`. */
export default function Score({ mode, value, compact = false }: { mode: Mode; value: number; compact?: boolean }) {
  const info = SCORE_INFO[mode];
  const tipId = useId();
  return (
    <span className={`${styles.wrap} ${compact ? styles.compact : ""}`}>
      <span className={styles.chip} tabIndex={0} aria-describedby={tipId}>
        <span className={styles.label}>{info.label}</span>
        <span className={styles.value}>{value.toFixed(info.digits)}</span>
      </span>
      <span role="tooltip" id={tipId} className={styles.tip}>
        <span className={styles.tipTitle}>Điểm {info.label}</span>
        <span className={styles.tipRange}>Khoảng giá trị: {info.range}</span>
        <span className={styles.tipBody}>{info.explain}</span>
        <span className={styles.tipNote}>
          Chỉ so sánh điểm giữa các kết quả của <b>cùng một câu hỏi, cùng một mode</b>. Giữa các mode, hãy so thứ hạng.
        </span>
      </span>
    </span>
  );
}
