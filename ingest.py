"""Encode the VieQuADRetrieval corpus with multilingual-e5-base and load it into Postgres."""

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from viequad_loader import load_corpus

DB_DSN = "postgresql://rag:rag@localhost:5432/rag"
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64


def make_passage(doc: dict) -> str:
    # e5 models expect an "passage: " prefix on indexed text (asymmetric query/passage encoding).
    title = (doc.get("title") or "").strip()
    text = doc["text"]
    return f"passage: {title}. {text}" if title else f"passage: {text}"


def main():
    corpus = load_corpus()
    doc_ids = list(corpus.keys())
    passages = [make_passage(corpus[doc_id]) for doc_id in doc_ids]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        passages, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE documents")
        for doc_id, embedding in zip(doc_ids, embeddings):
            doc = corpus[doc_id]
            cur.execute(
                "INSERT INTO documents (id, title, body, embedding) VALUES (%s, %s, %s, %s)",
                (doc_id, doc["title"] or None, doc["text"], embedding),
            )
    conn.close()
    print(f"Ingested {len(doc_ids)} documents")


if __name__ == "__main__":
    main()
