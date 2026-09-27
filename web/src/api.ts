// Typed client for the FastAPI backend (rag/api.py). Keep the shapes in sync with its pydantic models.

export type Mode = "bm25" | "dense" | "hybrid" | "hybrid_rerank";
export type Kind = "paragraph" | "span";

export const MODES: Mode[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];
export const MODE_LABEL: Record<Mode, string> = {
  bm25: "BM25",
  dense: "Dense",
  hybrid: "Hybrid (RRF)",
  hybrid_rerank: "Hybrid + Rerank",
};
export const KIND_LABEL: Record<Kind, string> = { paragraph: "đoạn văn", span: "cụm câu trả lời" };

export interface SearchResult {
  doc_id: string;
  score: number;
  title: string | null;
  body: string;
  kind: Kind;
}

export interface SearchResponse {
  mode: Mode;
  device: string;
  took_ms: number;
  retrieval_ms: number;
  rerank_ms: number;
  results: SearchResult[];
  // Set when the query is exactly a dataset question: its id and ground truth {doc_id: kind}.
  qid: string | null;
  judged: Record<string, Kind> | null;
}

export interface QueryItem {
  qid: string;
  text: string;
}

export interface QueryPage {
  total: number;
  items: QueryItem[];
}

export interface JudgedDoc {
  doc_id: string;
  kind: Kind;
  title: string | null;
  body: string;
}

export interface QueryDetail extends QueryItem {
  relevant: JudgedDoc[];
}

interface Describe {
  median: number;
}

export interface DatasetStats {
  n_docs: number;
  n_by_kind: Record<Kind, number>;
  n_articles: number;
  n_queries: number;
  query_length: Describe;
  doc_length: Record<Kind, Describe>;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export function search(q: string, mode: Mode, k: number): Promise<SearchResponse> {
  return getJSON(`/search?${new URLSearchParams({ q, mode, k: String(k) })}`);
}

export function datasetStats(): Promise<DatasetStats> {
  return getJSON("/dataset/stats");
}

export function datasetQueries(filter: string, offset: number, limit: number): Promise<QueryPage> {
  return getJSON(`/dataset/queries?${new URLSearchParams({ q: filter, offset: String(offset), limit: String(limit) })}`);
}

export function datasetQuery(qid: string): Promise<QueryDetail> {
  return getJSON(`/dataset/queries/${encodeURIComponent(qid)}`);
}

/** The answer-span text of a dataset question, used to highlight it inside its paragraph. */
export async function answerSpan(qid: string | null): Promise<string | null> {
  if (!qid) return null;
  const detail = await datasetQuery(qid);
  return detail.relevant.find((d) => d.kind === "span")?.body ?? null;
}

export const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err));
