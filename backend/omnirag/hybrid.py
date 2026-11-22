"""API-owned persistent Qdrant local index; SQLite remains the provenance authority."""

import hashlib
import logging
import threading
import time
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from omnirag.db import connection
from omnirag.encoders import LocalEncoders
from omnirag.retrieval import retrieve

log = logging.getLogger("omnirag.retrieval")


class RetrievalError(Exception):
    def __init__(self, code, message, status=503):
        self.code, self.message, self.status = code, message, status


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse ranks, not incomparable BM25/cosine scores. Stable ties use source IDs."""
    scores, rows = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, row in enumerate(ranking, 1):
            key = row["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            rows[key] = row
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    return [
        {**rows[key], "score": scores[key]}
        for key in sorted(scores, key=lambda key: (-scores[key], key))
    ]


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class HybridIndex:
    def __init__(self, settings, encoders=None):
        self.settings = settings
        self.encoders = encoders
        self.client = None
        self.collection = None
        self.error = None
        self.loaded = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def open(self):
        self.encoders = self.encoders or LocalEncoders(self.settings)
        self.collection = f"chunks_{self.encoders.signature}"
        # Exactly one API process owns this path. All access is serialized by self.lock.
        self.client = QdrantClient(
            path=str(self.settings.vectors),
            force_disable_check_same_thread=True,
        )
        fresh = not self.client.collection_exists(self.collection)
        if fresh:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.encoders.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        with connection(self.settings.database) as con:
            con.execute(
                """UPDATE vector_jobs SET status='pending', error=NULL
                WHERE status='processing' OR generation IS NULL OR generation != ? OR ?""",
                (self.collection, fresh),
            )
        self.loaded = True

    def start(self):
        self.thread = threading.Thread(target=self._run, name="vector-indexer", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with self.lock:
                self.open()
            while not self.stop_event.is_set():
                with self.lock:
                    worked = self.sync_one()
                if not worked:
                    self.stop_event.wait(1)
        except Exception:
            log.exception("Local retrieval unavailable")
            self.error = (
                "Retrieval setup failed. Check API logs and run scripts/prepare_retrieval.py."
            )
            self.loaded = False

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()  # Never close Qdrant while its owner is still writing.
        if self.client:
            self.client.close()

    def status(self):
        with connection(self.settings.database) as con:
            rows = con.execute("SELECT status, count(*) n FROM vector_jobs GROUP BY status")
            counts = {row["status"]: row["n"] for row in rows}
        return {
            "enabled": True,
            "loaded": self.loaded,
            "error": self.error,
            "documents": counts,
            "collection": self.collection,
        }

    def sync_one(self):
        with connection(self.settings.database) as con:
            con.execute("BEGIN IMMEDIATE")
            job = con.execute(
                """SELECT v.document_id FROM vector_jobs v JOIN documents d
                ON d.id=v.document_id WHERE v.status='pending' AND d.status='ready'
                ORDER BY d.created_at LIMIT 1"""
            ).fetchone()
            if not job:
                return False
            document_id = job["document_id"]
            con.execute(
                "UPDATE vector_jobs SET status='processing' WHERE document_id=?", (document_id,)
            )
            chunks = con.execute(
                "SELECT * FROM chunks WHERE document_id=? ORDER BY id", (document_id,)
            ).fetchall()
        try:
            # Delete-and-replace is replayable; the SQLite job is ready only after all writes.
            self.client.delete(
                self.collection,
                models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document_id)
                            )
                        ],
                    )
                ),
                wait=True,
            )
            batch = []
            for chunk in chunks:
                for ordinal, text in enumerate(self.encoders.windows(chunk["text"])):
                    batch.append((chunk, ordinal, text))
                    if len(batch) == 8:
                        self._upsert(batch)
                        batch = []
            if batch:
                self._upsert(batch)
            with connection(self.settings.database) as con:
                con.execute(
                    """UPDATE vector_jobs SET status='ready', generation=?, error=NULL,
                    updated_at=? WHERE document_id=? AND status='processing'""",
                    (self.collection, time.time(), document_id),
                )
        except Exception:
            log.exception("Vector indexing failed for %s", document_id)
            with connection(self.settings.database) as con:
                con.execute(
                    """UPDATE vector_jobs SET status='failed', error=?, updated_at=?
                    WHERE document_id=? AND status='processing'""",
                    (
                        "Vector indexing failed; check API logs and retry indexing.",
                        time.time(),
                        document_id,
                    ),
                )
        return True

    def _upsert(self, batch):
        vectors = self.encoders.embed([item[2] for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count mismatch")
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{chunk['id']}:{ordinal}")),
                vector=vector,
                payload={
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "text_hash": text_hash(chunk["text"]),
                },
            )
            for (chunk, ordinal, _), vector in zip(batch, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points, wait=True)

    def delete_document(self, document_id: str) -> None:
        """Remove a deleted document's vectors; best-effort so deletion never hangs."""
        if not (self.loaded and self.client and self.collection):
            return
        try:
            with self.lock:
                self.client.delete(
                    self.collection,
                    models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
        except Exception:
            log.exception("Vector cleanup failed for %s", document_id)

    def _scope(self, scope):
        with connection(self.settings.database) as con:
            sql = """SELECT d.id, v.status, v.generation FROM documents d
                LEFT JOIN vector_jobs v ON v.document_id=d.id WHERE d.status='ready'"""
            if scope:
                sql += f" AND d.id IN ({','.join('?' for _ in scope)})"
            rows = con.execute(sql, scope).fetchall()
        if any(row["status"] != "ready" or row["generation"] != self.collection for row in rows):
            raise RetrievalError(
                "INDEX_NOT_READY", "Semantic indexing is incomplete. Wait or select keyword search."
            )
        return [row["id"] for row in rows]

    def _dense(self, question, scope):
        ids = self._scope(scope)
        if not ids:
            return []
        try:
            vector = self.encoders.query(question)
        except ValueError as exc:
            raise RetrievalError("RETRIEVAL_INPUT", str(exc), 422) from exc
        hits = self.client.query_points(
            self.collection,
            query=vector,
            limit=self.settings.retrieval_candidates * 8,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=ids),
                    )
                ]
            ),
        ).points
        rows, seen = [], set()
        with connection(self.settings.database) as con:
            for hit in hits:
                key = hit.payload["chunk_id"]
                if key in seen:
                    continue
                row = con.execute(
                    """SELECT c.id chunk_id, c.document_id, c.page, c.text,
                    d.name document_name, COALESCE(p.method,'native') extraction_method
                    FROM chunks c JOIN documents d ON d.id=c.document_id
                    LEFT JOIN page_extractions p ON p.document_id=c.document_id AND p.page=c.page
                    WHERE c.id=? AND d.status='ready'""",
                    (key,),
                ).fetchone()
                if (
                    row
                    and row["document_id"] in ids
                    and text_hash(row["text"]) == hit.payload["text_hash"]
                ):
                    rows.append({**dict(row), "score": float(hit.score)})
                    seen.add(key)
        rows.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        return rows[: self.settings.retrieval_candidates]

    def search(self, question, scope, mode, limit):
        if not self.loaded:
            raise RetrievalError(
                "RETRIEVAL_UNAVAILABLE", self.error or "Retrieval models are loading."
            )
        if not self.lock.acquire(blocking=False):
            raise RetrievalError(
                "RETRIEVAL_BUSY", "Semantic search is busy indexing or searching.", 429
            )
        try:
            dense = self._dense(question, scope)
            if mode == "dense":
                return dense[:limit]
            lexical = retrieve(
                self.settings.database, question, scope, self.settings.retrieval_candidates
            )
            fused = reciprocal_rank_fusion([lexical, dense])[: self.settings.retrieval_candidates]
            if mode == "hybrid_rerank" and fused:
                scores = self.encoders.rerank(question, [row["text"] for row in fused])
                if len(scores) != len(fused):
                    raise ValueError("Reranking count mismatch")
                fused = [
                    {**row, "fusion_score": row["score"], "score": score}
                    for row, score in zip(fused, scores, strict=True)
                ]
                fused.sort(key=lambda row: (-row["score"], row["chunk_id"]))
            return fused[:limit]
        except RetrievalError:
            raise
        except Exception as exc:
            log.exception("Retrieval failed")
            raise RetrievalError(
                "RETRIEVAL_FAILED", "Semantic search failed. Check API logs."
            ) from exc
        finally:
            self.lock.release()
