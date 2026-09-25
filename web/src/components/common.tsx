// Small presentational pieces shared by both tabs.

import type { ReactNode } from "react";
import { KIND_LABEL, type Kind } from "../api";
import { CheckIcon, ChipIcon } from "./icons";
import styles from "./common.module.scss";

/** Coloured chip for a document kind (paragraph / answer span); `dot` renders just the swatch. */
export function KindTag({ kind, dot = false }: { kind: Kind; dot?: boolean }) {
  if (dot) return <span className={`${styles.dot} ${styles[kind]}`} title={KIND_LABEL[kind]} />;
  return (
    <span className={`${styles.kindTag} ${styles[kind]}`}>
      <span className={styles.dotInner} />
      {KIND_LABEL[kind]}
    </span>
  );
}

/** `text` with the first occurrence of `needle` wrapped in <mark>. */
export function Highlight({ text, needle }: { text: string; needle?: string | null }) {
  const i = needle ? text.indexOf(needle) : -1;
  if (!needle || i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark>{needle}</mark>
      {text.slice(i + needle.length)}
    </>
  );
}

export function Badge({ tone, children }: { tone: "good" | "warn" | "neutral"; children: ReactNode }) {
  return <span className={`${styles.badge} ${styles[tone]}`}>{children}</span>;
}

export function HitBadge() {
  return (
    <Badge tone="good">
      <CheckIcon size={12} strokeWidth={3} /> Đúng theo nhãn
    </Badge>
  );
}

export function DeviceBadge({ device }: { device: string }) {
  return (
    <Badge tone={device === "cuda" ? "good" : "warn"}>
      <ChipIcon size={12} /> {device.toUpperCase()}
    </Badge>
  );
}

/** One document card: used for search results and for a question's ground truth. */
export function DocCard({
  rank,
  title,
  tags,
  aside,
  body,
  hit = false,
  kind,
  delay = 0,
}: {
  rank?: number;
  title: ReactNode;
  tags?: ReactNode;
  aside?: ReactNode;
  body: ReactNode;
  hit?: boolean;
  kind: Kind;
  delay?: number;
}) {
  return (
    <article
      className={`${styles.doc} ${hit ? styles.hit : ""} ${kind === "span" ? styles.spanDoc : ""}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {rank !== undefined && <div className={styles.rank}>{rank}</div>}
      <div className={styles.docMain}>
        <header>
          <div className={styles.title}>{title}</div>
          {aside && <div className={styles.aside}>{aside}</div>}
        </header>
        {tags && <div className={styles.tags}>{tags}</div>}
        <div className={styles.body}>{body}</div>
      </div>
    </article>
  );
}

/** Placeholder cards while a search is running. */
export function SkeletonList({ count = 3 }: { count?: number }) {
  return (
    <div aria-busy="true" aria-label="Đang tải">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className={styles.skeleton}>
          <div className={styles.skRank} />
          <div className={styles.skMain}>
            <div className={styles.skLine} style={{ width: "40%" }} />
            <div className={styles.skLine} style={{ width: "95%" }} />
            <div className={styles.skLine} style={{ width: "80%" }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyIcon}>{icon}</div>
      <div className={styles.emptyTitle}>{title}</div>
      {children && <div className={styles.emptyText}>{children}</div>}
    </div>
  );
}
