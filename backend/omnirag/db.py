"""Single-node SQLite repository with explicit migrations; never use a network filesystem."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','processing','ready','failed')),
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    warnings TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_until REAL,
    lease_token TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(document_id, page, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(chunk_id UNINDEXED, body, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS query_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    response TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_status ON documents(status, created_at);
CREATE INDEX IF NOT EXISTS idx_chunk_document ON chunks(document_id);
"""

MIGRATION_2 = """
CREATE TABLE vector_jobs (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    generation TEXT,
    error TEXT,
    updated_at REAL NOT NULL DEFAULT 0
);
INSERT INTO vector_jobs(document_id) SELECT id FROM documents WHERE status='ready';
"""

MIGRATION_3 = """
CREATE TABLE page_extractions (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    method TEXT NOT NULL,
    metadata TEXT NOT NULL,
    PRIMARY KEY(document_id, page)
);
INSERT INTO page_extractions(document_id,page,method,metadata)
SELECT DISTINCT document_id,page,'native','{"legacy":true}' FROM chunks;
"""

MIGRATION_4 = """
CREATE TABLE document_summaries (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('overview','detailed')),
    status TEXT NOT NULL CHECK(status IN ('ready','failed')),
    content TEXT,
    coverage TEXT,
    model TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY(document_id, kind)
);
"""

MIGRATION_5 = """
CREATE TABLE assistant_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('coding','chat')),
    task TEXT NOT NULL,
    question TEXT NOT NULL,
    response TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assistant_runs_created ON assistant_runs(created_at);
"""

MIGRATION_5_ALTERS = [
    "ALTER TABLE documents ADD COLUMN source_url TEXT",
]


@contextmanager
def connection(path: Path):
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection(path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("BEGIN IMMEDIATE")
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version > 5:
            raise RuntimeError("Database schema is newer than this application.")
        if version == 0:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    con.execute(statement)
            version = 1
        if version == 1:
            for statement in MIGRATION_2.split(";"):
                if statement.strip():
                    con.execute(statement)
            version = 2
        if version == 2:
            for statement in MIGRATION_3.split(";"):
                if statement.strip():
                    con.execute(statement)
            version = 3
        if version == 3:
            for statement in MIGRATION_4.split(";"):
                if statement.strip():
                    con.execute(statement)
            version = 4
        if version == 4:
            for statement in MIGRATION_5.split(";"):
                if statement.strip():
                    con.execute(statement)
            for statement in MIGRATION_5_ALTERS:
                try:
                    con.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
        con.execute("PRAGMA user_version=5")


def public_document(row: sqlite3.Row) -> dict:
    fields = (
        "id",
        "name",
        "size_bytes",
        "created_at",
        "status",
        "page_count",
        "chunk_count",
        "error_code",
        "error_message",
        "attempts",
    )
    result = {key: row[key] for key in fields}
    result["warnings"] = json.loads(row["warnings"])
    keys = row.keys()
    result["source_url"] = row["source_url"] if "source_url" in keys else None
    return result


def get_document(path: Path, document_id: str):
    with connection(path) as con:
        return con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()


def document_chunks(path: Path, document_id: str) -> list[dict]:
    with connection(path) as con:
        rows = con.execute(
            "SELECT page, ordinal, text FROM chunks WHERE document_id=? ORDER BY page, ordinal",
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_summary(path: Path, document_id: str, kind: str):
    with connection(path) as con:
        row = con.execute(
            "SELECT * FROM document_summaries WHERE document_id=? AND kind=?",
            (document_id, kind),
        ).fetchone()
    return dict(row) if row else None


def save_summary(
    path: Path,
    document_id: str,
    kind: str,
    *,
    status: str,
    content: dict | None,
    coverage: dict | None,
    model: str | None,
    error: str | None,
) -> None:
    with connection(path) as con:
        con.execute(
            """INSERT INTO document_summaries
            (document_id,kind,status,content,coverage,model,error,created_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id,kind) DO UPDATE SET status=excluded.status,
            content=excluded.content, coverage=excluded.coverage, model=excluded.model,
            error=excluded.error, created_at=excluded.created_at""",
            (
                document_id,
                kind,
                status,
                json.dumps(content) if content is not None else None,
                json.dumps(coverage) if coverage is not None else None,
                model,
                error,
                time.time(),
            ),
        )


def summary_states(path: Path) -> dict[str, dict[str, str]]:
    """Latest per-document summary status for list views, keyed by document id."""
    with connection(path) as con:
        rows = con.execute(
            "SELECT document_id, kind, status, error FROM document_summaries"
        ).fetchall()
    states: dict[str, dict[str, str]] = {}
    for row in rows:
        states.setdefault(row["document_id"], {})[row["kind"]] = row["status"]
    return states


def claim_job(path: Path, timeout: int):
    now = time.time()
    with connection(path) as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """UPDATE documents SET status='failed', error_code='RETRY_EXHAUSTED',
            error_message='Worker recovery limit reached. Inspect the worker log.'
            WHERE status='processing' AND lease_until < ? AND attempts >= 3""",
            (now,),
        )
        row = con.execute(
            """SELECT * FROM documents WHERE attempts < 3 AND
            (status='queued' OR (status='processing' AND lease_until < ?))
            ORDER BY created_at LIMIT 1""",
            (now,),
        ).fetchone()
        if row is None:
            return None
        token = str(uuid4())
        con.execute(
            """UPDATE documents SET status='processing', attempts=attempts+1,
            lease_until=?, lease_token=? WHERE id=?""",
            (now + timeout + 30, token, row["id"]),
        )
        return {**dict(row), "lease_token": token}


def finish_job(path: Path, job: dict, result: dict) -> bool:
    with connection(path) as con:
        con.execute("BEGIN IMMEDIATE")
        current = con.execute(
            "SELECT lease_token, status FROM documents WHERE id=?", (job["id"],)
        ).fetchone()
        if not current or current["lease_token"] != job["lease_token"]:
            return False
        if current["status"] != "processing":
            return False
        if result.get("error_code"):
            con.execute(
                """UPDATE documents SET status='failed', error_code=?, error_message=?,
                lease_until=NULL, lease_token=NULL WHERE id=?""",
                (result["error_code"], result["error_message"], job["id"]),
            )
            return True
        # Replace both indexes atomically, including a recovered partially attempted job.
        con.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",
            (job["id"],),
        )
        con.execute("DELETE FROM chunks WHERE document_id=?", (job["id"],))
        con.execute("DELETE FROM page_extractions WHERE document_id=?", (job["id"],))
        for page in result.get("pages", []):
            con.execute(
                "INSERT INTO page_extractions VALUES(?,?,?,?)",
                (job["id"], page["page"], page["method"], json.dumps(page)),
            )
        for chunk in result["chunks"]:
            chunk_id = f"{job['id']}:{chunk['page']}:{chunk['ordinal']}"
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?)",
                (chunk_id, job["id"], chunk["page"], chunk["ordinal"], chunk["text"]),
            )
            con.execute("INSERT INTO chunks_fts VALUES(?,?)", (chunk_id, chunk["text"]))
        con.execute(
            """UPDATE documents SET status='ready', page_count=?, chunk_count=?, warnings=?,
            error_code=NULL, error_message=NULL, lease_until=NULL, lease_token=NULL WHERE id=?""",
            (
                result["page_count"],
                len(result["chunks"]),
                json.dumps(result["warnings"]),
                job["id"],
            ),
        )
        con.execute(
            """INSERT INTO vector_jobs(document_id) VALUES(?) ON CONFLICT(document_id)
            DO UPDATE SET status='pending', generation=NULL, error=NULL""",
            (job["id"],),
        )
        return True


def insert_ready_document(path: Path, document: dict, result: dict) -> None:
    """Store an already-extracted text document (DOCX/TXT/MD/CSV/JSON/URL) as ready.

    PDFs keep the leased worker path; these formats extract fast enough to ingest
    synchronously with the same chunk/FTS/vector pipeline and synthetic pages.
    """
    with connection(path) as con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute(
            "SELECT id FROM documents WHERE sha256=?", (document["sha256"],)
        ).fetchone()
        if existing:
            raise ValueError("duplicate")
        con.execute(
            "INSERT INTO documents(id,name,sha256,size_bytes,created_at,status,"
            "page_count,chunk_count,warnings,source_url) VALUES(?,?,?,?,?,'ready',?,?,?,?)",
            (
                document["id"],
                document["name"],
                document["sha256"],
                document["size_bytes"],
                time.time(),
                result["page_count"],
                len(result["chunks"]),
                json.dumps(result["warnings"]),
                document.get("source_url"),
            ),
        )
        for page in result["pages"]:
            con.execute(
                "INSERT INTO page_extractions VALUES(?,?,?,?)",
                (document["id"], page["page"], page["method"], json.dumps(page)),
            )
        for chunk in result["chunks"]:
            chunk_id = f"{document['id']}:{chunk['page']}:{chunk['ordinal']}"
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?)",
                (chunk_id, document["id"], chunk["page"], chunk["ordinal"], chunk["text"]),
            )
            con.execute("INSERT INTO chunks_fts VALUES(?,?)", (chunk_id, chunk["text"]))
        con.execute("INSERT INTO vector_jobs(document_id) VALUES(?)", (document["id"],))


def delete_document_everywhere(path: Path, document_id: str) -> dict:
    """Remove a document and all of its derived data; returns the row for cleanup."""
    with connection(path) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            return {}
        con.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE document_id=?)",
            (document_id,),
        )
        con.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM page_extractions WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM vector_jobs WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM document_summaries WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return dict(row)


def save_assistant_run(
    path: Path, run_id: str, kind: str, task: str, question: str, response: dict
) -> None:
    with connection(path) as con:
        con.execute(
            "INSERT INTO assistant_runs VALUES(?,?,?,?,?,?)",
            (run_id, time.time(), kind, task, question, json.dumps(response)),
        )


def assistant_history(path: Path, limit: int = 20) -> list[dict]:
    with connection(path) as con:
        rows = con.execute(
            "SELECT response FROM assistant_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(row[0]) for row in rows]
