import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, closing
from typing import Literal
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from omnirag.auth import AuthStore, install_auth
from omnirag.coding import (
    MAX_CODE_CHARACTERS,
    MAX_ERROR_CHARACTERS,
    CodingService,
    build_gateway,
)
from omnirag.config import Settings
from omnirag.db import (
    connection,
    delete_document_everywhere,
    document_chunks,
    get_document,
    get_summary,
    initialize,
    insert_ready_document,
    public_document,
    save_assistant_run,
    save_summary,
    summary_states,
)
from omnirag.generation import (
    COST_WORDS,
    GenerationError,
    OllamaGateway,
    _question_asks_cost,
    is_personal_record_question,
    validate_citations,
    validate_grounded,
    verify_individual_record,
)
from omnirag.hybrid import HybridIndex, RetrievalError
from omnirag.ingestion import (
    IngestionError,
    fetch_url_text,
    parse_docx,
    parse_text_document,
    parse_web_page,
)
from omnirag.intents import OVERVIEW_KINDS, classify_intent, is_action_request
from omnirag.retrieval import retrieve
from omnirag.security import LocalRequestGuard
from omnirag.summary import coverage_sample, generate_overview, partial_note

log = logging.getLogger("omnirag.api")


class ConversationTurn(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    answer: str = Field(default="", max_length=1200)
    document_ids: list[UUID] = Field(default_factory=list, max_length=5)


class CodingTurn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(default="", max_length=4000)


class ChatInput(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    conversation: list[CodingTurn] = Field(default_factory=list, max_length=2)


class CodingInput(ChatInput):
    task: Literal["generate", "explain", "debug", "improve", "review", "auto"] = "auto"
    code: str = Field(default="", max_length=MAX_CODE_CHARACTERS)
    error: str = Field(default="", max_length=MAX_ERROR_CHARACTERS)
    language: str = Field(default="", max_length=40)
    document_ids: list[UUID] = Field(default_factory=list, max_length=10)


class UrlInput(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


class QueryInput(BaseModel):
    question: str = Field(min_length=3, max_length=1200)
    document_ids: list[UUID] = Field(default_factory=list, max_length=30)
    retrieval_mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] | None = None
    conversation: list[ConversationTurn] = Field(default_factory=list, max_length=2)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("Enter at least three non-whitespace characters.")
        return value.strip()


class SummaryInput(BaseModel):
    kind: Literal["overview", "detailed"] = "overview"


def fail(status: int, code: str, message: str):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def create_app(
    settings: Settings | None = None, gateway=None, index=None, coding_gateway=None
) -> FastAPI:
    settings = settings or Settings()
    gateway = gateway or OllamaGateway(settings)
    coding_gateway = coding_gateway or build_gateway(settings)
    coding_service = CodingService(settings, coding_gateway)
    generation_slot = threading.BoundedSemaphore(1)
    auth_store = AuthStore(settings.data_dir.resolve() / "auth")
    index = index or (HybridIndex(settings) if settings.retrieval_enabled else None)
    grounded = settings.answer_mode == "grounded" and hasattr(gateway, "generate_grounded")

    @asynccontextmanager
    async def lifespan(_app):
        settings.prepare()
        initialize(settings.database)
        if settings.auth_enabled:
            auth_store.initialize()
        if index:
            index.start()
        try:
            yield
        finally:
            if index:
                index.close()

    app = FastAPI(title="OmniRAG AI", version="0.4.0", lifespan=lifespan)
    install_auth(app, settings, auth_store)
    app.add_middleware(LocalRequestGuard, max_bytes=settings.upload_max_mb * 1024 * 1024 + 65536)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.exception_handler(sqlite3.OperationalError)
    async def database_error(_request: Request, _exc: sqlite3.OperationalError):
        log.exception("Database operation failed")
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "DATABASE_UNAVAILABLE",
                    "message": "Database unavailable. Check disk and worker logs.",
                }
            },
        )

    @app.get("/api/v1/health")
    def health():
        with connection(settings.database) as con:
            con.execute("SELECT count(*) FROM documents").fetchone()
        return {
            "status": "ok",
            "phase": 13,
            "retrieval": settings.default_retrieval_mode,
            "authentication": settings.auth_enabled,
            "answer_mode": settings.answer_mode,
            "coding_provider": settings.code_provider,
        }

    @app.get("/api/v1/readiness")
    def readiness():
        model = gateway.ready()
        with connection(settings.database) as con:
            heartbeat = con.execute(
                "SELECT value FROM runtime WHERE key='worker_heartbeat'"
            ).fetchone()
        worker = bool(
            heartbeat and time.time() - heartbeat[0] < settings.ingest_timeout_seconds + 30
        )
        return {
            "model": model,
            "worker_active": worker,
            "local_only": True,
            "retrieval": index.status() if index else {"enabled": False, "loaded": False},
            "default_retrieval_mode": settings.default_retrieval_mode,
            "ocr": {"enabled": settings.ocr_enabled, "language": "eng"},
            "answer_mode": settings.answer_mode,
            "coding": {"provider": settings.code_provider, **coding_gateway.ready()},
        }

    @app.get("/api/v1/documents")
    def documents():
        with connection(settings.database) as con:
            rows = con.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            jobs = {
                row["document_id"]: dict(row)
                for row in con.execute("SELECT document_id, status, error FROM vector_jobs")
            }
            summaries = summary_states(settings.database)
            results = []
            for row in rows:
                kind = (
                    "url"
                    if row["source_url"]
                    else "pdf"
                    if (settings.uploads / f"{row['id']}.pdf").is_file()
                    else "text"
                )
                results.append(
                    {
                        **public_document(row),
                        "kind": kind,
                        "vector_status": jobs.get(row["id"], {}).get("status"),
                        "vector_error": jobs.get(row["id"], {}).get("error"),
                        "summary_status": summaries.get(row["id"], {}).get("overview"),
                        "detailed_summary_status": summaries.get(row["id"], {}).get("detailed"),
                    }
                )
            return results

    TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}

    @app.post("/api/v1/documents", status_code=202)
    async def upload(file: UploadFile):
        original = (file.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
        name = re.sub(r"[\x00-\x1f\x7f]", "", original)[:180]
        suffix = f".{name.rsplit('.', 1)[1].lower()}" if "." in name else ""
        if suffix != ".pdf" and suffix not in TEXT_SUFFIXES and suffix != ".docx":
            await file.close()
            fail(
                415,
                "UNSUPPORTED_TYPE",
                "Supported formats: PDF, DOCX, TXT, Markdown, CSV and JSON.",
            )
        document_id = str(uuid4())
        temporary = settings.uploads / f"{document_id}.part"
        destination = settings.uploads / f"{document_id}.pdf"
        digest, size, prefix = hashlib.sha256(), 0, b""
        stored = False
        try:
            with temporary.open("xb") as target:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > settings.upload_max_mb * 1024 * 1024:
                        fail(
                            413,
                            "UPLOAD_TOO_LARGE",
                            "The file exceeds the configured upload limit.",
                        )
                    if not prefix:
                        prefix = block[:5]
                    digest.update(block)
                    target.write(block)
            if suffix == ".pdf":
                if prefix != b"%PDF-":
                    fail(
                        415,
                        "INVALID_PDF_HEADER",
                        "The uploaded file does not have a PDF signature.",
                    )
                with connection(settings.database) as con:
                    con.execute("BEGIN IMMEDIATE")
                    existing = con.execute(
                        "SELECT * FROM documents WHERE sha256=?", (digest.hexdigest(),)
                    ).fetchone()
                    if existing:
                        return {"document": public_document(existing), "duplicate": True}
                    os.replace(temporary, destination)
                    con.execute(
                        "INSERT INTO documents(id,name,sha256,size_bytes,created_at,status) "
                        "VALUES(?,?,?,?,?,'queued')",
                        (document_id, name, digest.hexdigest(), size, time.time()),
                    )
                stored = True
                return {
                    "document": public_document(get_document(settings.database, document_id)),
                    "duplicate": False,
                }
            # Text formats extract synchronously and enter the same chunk/FTS/vector
            # pipeline as ready PDFs, with synthetic pages for page-linked citations.
            if size == 0:
                fail(422, "EMPTY_DOCUMENT", "The uploaded file is empty.")
            try:
                if suffix == ".docx":
                    result = parse_docx(temporary)
                else:
                    result = parse_text_document(temporary, suffix)
            except IngestionError as exc:
                fail(422, exc.code, exc.message)
            record = {
                "id": document_id,
                "name": name,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
            try:
                insert_ready_document(settings.database, record, result)
            except ValueError:
                with connection(settings.database) as con:
                    existing = con.execute(
                        "SELECT * FROM documents WHERE sha256=?", (digest.hexdigest(),)
                    ).fetchone()
                if existing is None:
                    fail(502, "INGEST_FAILED", "The document could not be stored.")
                return {"document": public_document(existing), "duplicate": True}
            return {
                "document": public_document(get_document(settings.database, document_id)),
                "duplicate": False,
            }
        finally:
            temporary.unlink(missing_ok=True)
            if not stored:
                destination.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/v1/documents/{document_id}")
    def document(document_id: UUID):
        row = get_document(settings.database, str(document_id))
        if row is None:
            fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        with connection(settings.database) as con:
            job = con.execute(
                "SELECT status, error FROM vector_jobs WHERE document_id=?", (str(document_id),)
            ).fetchone()
            pages = [
                {**json.loads(p["metadata"]), "page": p["page"], "method": p["method"]}
                for p in con.execute(
                    "SELECT * FROM page_extractions WHERE document_id=? ORDER BY page",
                    (str(document_id),),
                )
            ]
        pdf_path = settings.uploads / f"{document_id}.pdf"
        kind = (
            "url" if row["source_url"] else "pdf" if pdf_path.is_file() else "text"
        )
        return {
            **public_document(row),
            "kind": kind,
            "vector_status": job["status"] if job else None,
            "vector_error": job["error"] if job else None,
            "pages": pages,
        }

    @app.delete("/api/v1/documents/{document_id}")
    def delete_document(document_id: UUID):
        """Remove a source and its derived data from every store, or nothing."""
        document_id = str(document_id)
        row = delete_document_everywhere(settings.database, document_id)
        if not row:
            fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        (settings.uploads / f"{document_id}.pdf").unlink(missing_ok=True)
        (settings.uploads / f"{document_id}.part").unlink(missing_ok=True)
        if index:
            index.delete_document(document_id)
        return {"deleted": True, "id": document_id}

    @app.post("/api/v1/documents/{document_id}/retry", status_code=202)
    def retry_document(document_id: UUID):
        # Ready documents are immutable: replacing their chunks would invalidate saved citations.
        with connection(settings.database) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM documents WHERE id=?", (str(document_id),)).fetchone()
            if row is None:
                fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
            if row["status"] != "failed":
                fail(409, "DOCUMENT_NOT_FAILED", "Only failed documents can be retried.")
            con.execute(
                """UPDATE documents SET status='queued', attempts=0, error_code=NULL,
                error_message=NULL, lease_token=NULL, lease_until=NULL WHERE id=?""",
                (str(document_id),),
            )
        return {"status": "queued"}

    @app.get("/api/v1/documents/{document_id}/pages/{page}/preview")
    def page_preview(document_id: UUID, page: int):
        """Rasterized single-page preview the in-app viewer can always render.

        Headless and embedded PDF viewers are unreliable; this renders the page
        with the local PDFium build into a bounded PNG. Original bytes stay local.
        """
        row = get_document(settings.database, str(document_id))
        path = settings.uploads / f"{document_id}.pdf"
        if row is None or not path.is_file():
            fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        if row["status"] != "ready" or not 1 <= page <= (row["page_count"] or 0):
            fail(422, "PAGE_OUT_OF_RANGE", "That page is not part of this document.")
        import pypdfium2 as pdfium

        try:
            with pdfium.PdfDocument(path) as document:
                if page - 1 >= len(document):
                    fail(422, "PAGE_OUT_OF_RANGE", "That page is not part of this document.")
                with closing(document[page - 1]) as pdf_page:
                    width, height = pdf_page.get_size()
                    if not all(math.isfinite(n) and n > 0 for n in (width, height)):
                        fail(422, "INVALID_PAGE_SIZE", "Invalid PDF page dimensions.")
                    # Bound both dimensions before PDFium allocates the bitmap.
                    scale = min(1100 / width, 2000 / height, 1100 / 612)
                    if min(width * scale, height * scale) < 1:
                        fail(422, "INVALID_PAGE_SIZE", "PDF page is too elongated to preview.")
                    with closing(pdf_page.render(scale=scale)) as bitmap:
                        with bitmap.to_pil() as image, image.convert("RGB") as rgb:
                            buffer = io.BytesIO()
                            rgb.save(buffer, format="PNG")
        except HTTPException:
            raise
        except Exception:
            log.exception("Page preview failed for %s page %s", document_id, page)
            fail(502, "PREVIEW_FAILED", "The page could not be rendered locally.")
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/v1/documents/{document_id}/file")
    def document_file(document_id: UUID):
        row = get_document(settings.database, str(document_id))
        path = settings.uploads / f"{document_id}.pdf"
        if row is None or not path.is_file():
            fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=row["name"],
            content_disposition_type="inline",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    def check_scope(document_ids: list[str]) -> None:
        for document_id in document_ids:
            doc = get_document(settings.database, document_id)
            if doc is None:
                fail(404, "DOCUMENT_NOT_FOUND", "A selected document does not exist.")
            if doc["status"] == "failed":
                fail(
                    409,
                    "DOCUMENT_NOT_READY",
                    f"“{doc['name']}” failed to process"
                    + (f" ({doc['error_message']})" if doc["error_message"] else "")
                    + ". Retry it from the document list or pick another document.",
                )
            if doc["status"] != "ready":
                fail(
                    409,
                    "DOCUMENT_NOT_READY",
                    f"“{doc['name']}” is still processing. Wait for it to finish "
                    "or pick a ready document.",
                )

    def ready_documents() -> list[dict]:
        with connection(settings.database) as con:
            rows = con.execute(
                "SELECT id, name FROM documents WHERE status='ready' ORDER BY created_at"
            ).fetchall()
        return [{"id": row["id"], "name": row["name"]} for row in rows]

    def search_evidence(
        body: QueryInput, scope: list[str], limit: int | None = None, question: str | None = None
    ):
        mode = body.retrieval_mode or settings.default_retrieval_mode
        check_scope(scope)
        effective_limit = limit or settings.top_k
        effective_question = question or body.question
        if mode == "bm25":
            return (
                retrieve(settings.database, effective_question, scope, effective_limit),
                mode,
            )
        if not index:
            fail(503, "RETRIEVAL_DISABLED", "Semantic search is disabled. Select keyword search.")
        try:
            return index.search(
                effective_question, scope, mode, effective_limit
            ), mode
        except RetrievalError as exc:
            fail(exc.status, exc.code, exc.message)

    def coverage_evidence(scope: list[str]) -> tuple[list[dict], dict]:
        """Reading-order evidence for whole-document questions; no similarity ranking."""
        check_scope(scope)
        rows, used_pages = [], set()
        total_chunks = 0
        for document_id in scope:
            doc = get_document(settings.database, document_id)
            chunks = document_chunks(settings.database, document_id)
            total_chunks += len(chunks)
            sample, _ = coverage_sample(chunks)
            for chunk in sample:
                rows.append(
                    {
                        "chunk_id": f"{document_id}:{chunk['page']}:{chunk['ordinal']}",
                        "document_id": document_id,
                        "document_name": doc["name"],
                        "page": chunk["page"],
                        "ordinal": chunk["ordinal"],
                        "text": chunk["text"],
                        "extraction_method": "native",
                        "selection": "coverage",
                    }
                )
                used_pages.add(chunk["page"])
        pages_total = 0
        with connection(settings.database) as con:
            for document_id in scope:
                pages_total += (
                    con.execute(
                        "SELECT page_count FROM documents WHERE id=?", (document_id,)
                    ).fetchone()["page_count"]
                    or 0
                )
        coverage = {
            "complete": len(used_pages) >= pages_total and total_chunks > 0,
            "pages_used": len(used_pages),
            "pages_total": pages_total,
        }
        return rows, coverage

    def run_generation(question, evidence, conversation, intent=None, scope=None):
        turns = [
            {"question": turn.question, "answer": turn.answer}
            for turn in (conversation or [])
            # Context reset at document boundaries: a follow-up answered from
            # another document must not steer this one.
            if not scope
            or not turn.document_ids
            or {str(item) for item in turn.document_ids} & set(scope)
        ]
        if grounded:
            return gateway.generate_grounded(
                question,
                evidence,
                conversation=turns,
                intent=intent,
            )
        return gateway.generate(question, evidence)

    def build_claims(draft, evidence, question="", attribute=""):
        if grounded:
            return validate_grounded(draft, evidence, question, attribute)
        return validate_citations(
            draft, evidence, extractive=getattr(gateway, "answer_strategy", None)
            == "extractive_source_binding"
        ), {}

    def summary_result(document_id, doc_name, content, coverage, cached, started, question):
        claims = []
        with connection(settings.database) as con:
            chunk_index = {
                row["page"]: row["id"]
                for row in con.execute(
                    "SELECT id, page FROM chunks WHERE document_id=? ORDER BY page, ordinal",
                    (document_id,),
                )
            }
        for point in content.get("key_points", []):
            page = point["pages"][0] if point.get("pages") else None
            claims.append(
                {
                    "text": point["text"],
                    "kind": "explanation",
                    "citations": [
                        {
                            "chunk_id": chunk_index.get(page, ""),
                            "document_id": document_id,
                            "document_name": doc_name,
                            "extraction_method": "native",
                            "page": page or 1,
                            "quote": "",
                        }
                    ]
                    if page
                    else [],
                }
            )
        note = partial_note(coverage)
        result = {
            "id": str(uuid4()),
            "question": question,
            "status": "answered",
            "reason": (
                "Document summary generated from indexed text; key points link to their pages."
            ),
            "summary": content,
            "claims": claims,
            "evidence": [],
            "verification": "document_summary",
            "retrieval_mode": "summary",
            "retrieval_collection": None,
            "model": settings.ollama_model if not cached else cached.get("model"),
            "timings": {
                "retrieval_ms": 0,
                "generation_ms": round((time.perf_counter() - started) * 1000, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "usage": {"summary_kind": "cached" if cached else content.get("kind", "overview")},
            "scope": {"document_ids": [document_id], "source": "selection"},
            "coverage": coverage,
        }
        if note:
            result["reason"] = f"Partial coverage. {note}"
        return result

    def persist(result):
        with connection(settings.database) as con:
            con.execute(
                "INSERT INTO query_runs VALUES(?,?,?,?)",
                (result["id"], time.time(), result["question"], json.dumps(result)),
            )

    def summarize_document(document_id, kind, started, question):
        doc = get_document(settings.database, document_id)
        if doc is None:
            fail(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        if doc["status"] != "ready":
            fail(409, "DOCUMENT_NOT_READY", "Wait for the document to finish processing.")
        if not generation_slot.acquire(blocking=False):
            fail(429, "MODEL_BUSY", "Another answer or summary is running. Try again shortly.")
        try:
            chunks = document_chunks(settings.database, document_id)
            if not chunks:
                fail(409, "SUMMARY_EMPTY", "This document has no indexed text to summarize.")
            if kind == "detailed":
                from omnirag.summary import generate_detailed

                content, coverage = generate_detailed(gateway, doc["name"], chunks)
            else:
                content, coverage = generate_overview(gateway, doc["name"], chunks)
            save_summary(
                settings.database,
                document_id,
                kind,
                status="ready",
                content=content,
                coverage=coverage,
                model=settings.ollama_model,
                error=None,
            )
            result = summary_result(
                document_id, doc["name"], content, coverage, None, started, question
            )
            result["summary"]["kind"] = kind
            persist(result)
            return result
        except GenerationError as exc:
            save_summary(
                settings.database,
                document_id,
                kind,
                status="failed",
                content=None,
                coverage=None,
                model=None,
                error=f"{exc.code}: {exc.message}",
            )
            raise
        finally:
            generation_slot.release()

    @app.post("/api/v1/documents/{document_id}/summary")
    def create_summary(document_id: UUID, body: SummaryInput):
        try:
            return summarize_document(
                str(document_id), body.kind, time.perf_counter(), f"Summarize {body.kind}"
            )
        except GenerationError as exc:
            fail(exc.status_code, exc.code, exc.message)

    @app.get("/api/v1/documents/{document_id}/summary")
    def read_summary(document_id: UUID, kind: Literal["overview", "detailed"] = "overview"):
        row = get_summary(settings.database, str(document_id), kind)
        if row is None:
            fail(
                404,
                "SUMMARY_NOT_FOUND",
                "No summary has been generated yet. Choose Summarize document.",
            )
        doc = get_document(settings.database, str(document_id))
        return {
            "document_id": str(document_id),
            "document_name": doc["name"] if doc else None,
            "kind": kind,
            "status": row["status"],
            "content": json.loads(row["content"]) if row["content"] else None,
            "coverage": json.loads(row["coverage"]) if row["coverage"] else None,
            "model": row["model"],
            "error": row["error"],
            "created_at": row["created_at"],
        }

    @app.post("/api/v1/search")
    def search(body: QueryInput):
        started = time.perf_counter()
        intent = classify_intent(body.question)
        scope = [str(item) for item in dict.fromkeys(body.document_ids)]
        evidence, mode = search_evidence(body, scope)
        return {
            "evidence": evidence,
            "retrieval_mode": mode,
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 2),
            "collection": index.collection if index and mode != "bm25" else None,
            "intent": intent["kind"],
        }

    @app.post("/api/v1/retrieval/retry")
    def retry_vectors():
        with connection(settings.database) as con:
            count = con.execute(
                "UPDATE vector_jobs SET status='pending', error=NULL WHERE status='failed'"
            ).rowcount
        return {"queued": count}

    # --- AI coding assistant and general chat (plain-text generation workflows) ---

    def coding_sources(passages: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": passage["chunk_id"],
                "document_id": passage["document_id"],
                "document_name": passage["document_name"],
                "page": passage["page"],
                "text": passage["text"][:600],
            }
            for passage in passages
        ]

    def run_assistant(kind: str, body: CodingInput | ChatInput, task: str):
        started = time.perf_counter()
        question = body.question.strip()
        passages: list[dict] = []
        scope = []
        if kind == "coding" and isinstance(body, CodingInput) and body.document_ids:
            scope = [str(item) for item in dict.fromkeys(body.document_ids)]
            check_scope(scope)
            passages = retrieve(settings.database, question, scope, settings.top_k)
        if not generation_slot.acquire(blocking=False):
            fail(429, "MODEL_BUSY", "Another answer is running. Please try again shortly.")
        try:
            result = coding_service.assist(
                question,
                task,
                code=getattr(body, "code", ""),
                error=getattr(body, "error", ""),
                language=getattr(body, "language", ""),
                conversation=[turn.model_dump() for turn in body.conversation],
                passages=passages,
            )
        except GenerationError as exc:
            fail(exc.status_code, exc.code, exc.message)
        finally:
            generation_slot.release()
        run_id = str(uuid4())
        model = (
            settings.openrouter_model
            if settings.code_provider == "openrouter"
            else settings.ollama_code_model or settings.ollama_model
        )
        response = {
            "id": run_id,
            "kind": kind,
            "task": result["task"],
            "question": question,
            "status": "answered",
            "answer": result["answer"],
            "language": result["language"],
            "grounded": result["grounded"],
            "sources": coding_sources(passages) if result["grounded"] else [],
            "model": model,
            "verification": "assistant_generated",
            "retrieval_mode": "bm25" if result["grounded"] else None,
            "retrieval_collection": None,
            "timings": {
                "retrieval_ms": 0,
                "generation_ms": round((time.perf_counter() - started) * 1000, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }
        save_assistant_run(settings.database, run_id, kind, result["task"], question, response)
        return response

    @app.post("/api/v1/coding/assist")
    def coding_assist(body: CodingInput):
        task = body.task if body.task != "auto" else "auto"
        return run_assistant("coding", body, task)

    @app.post("/api/v1/assistant/chat")
    def assistant_chat(body: ChatInput):
        return run_assistant("chat", body, "chat")

    # --- Web page ingestion (ported from the RAG project, with SSRF guards) ---

    @app.post("/api/v1/urls", status_code=202)
    def ingest_url(body: UrlInput):
        try:
            text, title, final_url = fetch_url_text(
                body.url,
                settings.url_fetch_timeout_seconds,
                settings.url_fetch_max_mb,
            )
            result = parse_web_page(text)
        except IngestionError as exc:
            fail(422, exc.code, exc.message)
        except httpx.HTTPStatusError as exc:
            fail(
                502,
                "URL_FETCH_FAILED",
                f"The page responded with HTTP {exc.response.status_code}.",
            )
        except httpx.HTTPError:
            fail(502, "URL_FETCH_FAILED", "The page could not be fetched.")
        document_id = str(uuid4())
        name = re.sub(r"[\x00-\x1f\x7f]", "", title or final_url)[:180] or "Web page"
        record = {
            "id": document_id,
            "name": name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "size_bytes": len(text.encode()),
            "source_url": final_url,
        }
        try:
            insert_ready_document(settings.database, record, result)
        except ValueError:
            with connection(settings.database) as con:
                existing = con.execute(
                    "SELECT * FROM documents WHERE sha256=?", (record["sha256"],)
                ).fetchone()
            if existing is None:
                fail(502, "INGEST_FAILED", "The page could not be stored.")
            return {"document": public_document(existing), "duplicate": True}
        return {"document": public_document(get_document(settings.database, document_id))}

    @app.post("/api/v1/queries")
    def query(body: QueryInput):
        started = time.perf_counter()
        intent = classify_intent(body.question)
        scope = [str(item) for item in dict.fromkeys(body.document_ids)]
        scope_source = "selection"

        # Follow-up pronoun resolution: a conversation turn may carry its document scope.
        if not scope and body.conversation:
            for turn in reversed(body.conversation):
                if turn.document_ids:
                    scope = [str(item) for item in turn.document_ids]
                    scope_source = "conversation"
                    break

        # Whole-document questions resolve "this document" when exactly one is ready.
        if not scope and intent["kind"] in OVERVIEW_KINDS:
            ready = ready_documents()
            if not ready:
                fail(
                    409,
                    "NO_READY_DOCUMENTS",
                    "Upload a PDF and wait for it to finish processing first.",
                )
            if len(ready) == 1:
                scope, scope_source = [ready[0]["id"]], "single_document"
            else:
                result = {
                    "id": str(uuid4()),
                    "question": body.question,
                    "status": "clarification_needed",
                    "reason": (
                        "Your question refers to a document, and several documents are "
                        "ready. Choose which one to use."
                    ),
                    "claims": [],
                    "evidence": [],
                    "candidates": ready,
                    "verification": "clarification",
                    "retrieval_mode": body.retrieval_mode
                    or settings.default_retrieval_mode,
                    "retrieval_collection": None,
                    "model": None,
                    "timings": {
                        "retrieval_ms": 0,
                        "generation_ms": 0,
                        "total_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                    "usage": {"intent": intent["kind"]},
                    "scope": {"document_ids": [], "source": "none"},
                }
                persist(result)
                return result

        # Explicit summaries use the cached or freshly generated document overview.
        if intent["kind"] in {"summarize", "key_points"} and len(scope) == 1:
            cached = get_summary(settings.database, scope[0], "overview")
            if cached and cached["status"] == "ready":
                doc = get_document(settings.database, scope[0])
                result = summary_result(
                    scope[0],
                    doc["name"],
                    json.loads(cached["content"]),
                    json.loads(cached["coverage"]),
                    cached,
                    started,
                    body.question,
                )
                persist(result)
                return result
            try:
                return summarize_document(scope[0], "overview", started, body.question)
            except GenerationError as exc:
                fail(exc.status_code, exc.code, exc.message)

        is_overview = intent["kind"] in OVERVIEW_KINDS
        if not is_overview and is_action_request(body.question):
            result = {
                "id": str(uuid4()),
                "question": body.question,
                "status": "unsupported_question",
                "reason": (
                    "This looks like an instruction rather than a question about a "
                    "document. I can only answer questions using the indexed text, "
                    "and I do not perform actions."
                ),
                "claims": [],
                "evidence": [],
                "verification": "grounded_citation_v1" if grounded else "extractive_source_binding",
                "retrieval_mode": body.retrieval_mode or settings.default_retrieval_mode,
                "retrieval_collection": None,
                "model": None,
                "timings": {
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "total_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                "usage": {"intent": intent["kind"], "action_request": True},
                "scope": {"document_ids": scope, "source": scope_source},
                "intent": intent["kind"],
            }
            persist(result)
            return result
        limit = min(settings.top_k * 2, 8) if intent["kind"] == "list" else None
        coverage = None
        # Follow-up questions such as "how does it work?" retrieve nothing on their
        # own. The previous exchange supplies the subject; this expands retrieval
        # with the user's own words only, never with synonyms or outside terms.
        retrieval_question = body.question
        if body.conversation and not is_overview:
            last = body.conversation[-1]
            retrieval_question = (
                f"{body.question} {last.question} {last.answer[:300]}".strip()
            )
            usage_expanded = True
        else:
            usage_expanded = False
        if is_overview:
            evidence, coverage = coverage_evidence(scope)
            mode = "coverage"
        else:
            evidence, mode = search_evidence(
                body, scope, limit, question=retrieval_question
            )
        retrieval_ms = (time.perf_counter() - started) * 1000
        claims, usage, status = [], {}, "no_relevant_passage"
        reason, suggestion, missing_information = "", "", ""
        generation_ms = 0.0
        if evidence:
            if not generation_slot.acquire(blocking=False):
                fail(429, "MODEL_BUSY", "Another answer is running. Please try again shortly.")
            generation_started = time.perf_counter()
            try:
                for validation_attempt in range(2):
                    draft, usage = run_generation(
                        body.question, evidence, body.conversation, intent["kind"], scope
                    )
                    try:
                        claims, claim_usage = build_claims(
                            draft, evidence, body.question, usage.get("asked_attribute", "")
                        )
                        usage.update(claim_usage)
                        break
                    except GenerationError as exc:
                        if (
                            validation_attempt == 0
                            and exc.code in {"INVALID_CITATION", "MODEL_INVALID_OUTPUT"}
                        ):
                            # One bounded regeneration when the first response failed
                            # source validation; repeated failures still withhold.
                            usage["validation_retry"] = True
                            continue
                        raise
                if usage.get("attribute_absent"):
                    status = "unsupported_question"
                    claims = []
                    reason = (
                        f"The document never mentions “{usage.get('asked_attribute', '')}”, "
                        "which is what the question asks about; the retrieved facts are "
                        "about something else."
                    )
                    suggestion = (
                        "Check whether the right document is selected, or ask about "
                        "what the document does contain."
                    )
                elif usage.get("asked_record_absent"):
                    status = "unsupported_question"
                    claims = []
                    reason = (
                        "The document has no record of that exact identifier; the "
                        "retrieved passages concern other records."
                    )
                    suggestion = (
                        "Check the identifier, or confirm the right document is selected."
                    )
                elif grounded and usage.get("record_gate_withheld"):
                    status = "unsupported_question"
                    claims = []
                    reason = (
                        "The question asks about a specific person, order or event, and "
                        "the document contains no record of it; a general policy cannot "
                        "establish that it happened."
                    )
                    suggestion = (
                        "If a record of this case exists, check that the right document "
                        "is selected."
                    )
                elif draft.answerable and claims:
                    if (
                        grounded
                        and is_personal_record_question(body.question)
                        and not verify_individual_record(
                            gateway,
                            body.question,
                            evidence,
                            time.monotonic() + settings.model_timeout_seconds,
                        )["record_present"]
                    ):
                        # A second bounded pass guards the policy/record distinction:
                        # a general policy never proves an individual event.
                        status = "unsupported_question"
                        claims = []
                        reason = (
                            "The document has no explicit record of that specific "
                            "case; the retrieved text only contains general rules."
                        )
                        suggestion = (
                            "If a record of this case exists, check that the right "
                            "document is selected."
                        )
                    elif grounded and _question_asks_cost(body.question) and not any(
                        word in " ".join(claim["text"] for claim in claims).casefold()
                        for word in COST_WORDS
                    ):
                        # The requested attribute is a cost, but no kept claim states
                        # one: a duration or policy answer would substitute attributes.
                        status = "unsupported_question"
                        claims = []
                        reason = (
                            "The document does not state a cost or price for that; the "
                            "retrieved text answers a different question."
                        )
                        suggestion = (
                            "Check whether the document really contains pricing "
                            "information, or ask about what it does state."
                        )
                    else:
                        status = "answered"
                        reason = (
                            "Written from the cited passages; quotes were verified "
                            "against the indexed text."
                            if grounded
                            else "Selected source excerpts; compare them with the "
                            "original document."
                        )
                        if usage.get("claims_dropped"):
                            reason += (
                                f" {usage['claims_dropped']} model statement(s) were "
                                "withheld because the cited text did not support them."
                            )
                elif draft.answerable:
                    status = "unsupported_question"
                    reason = (
                        "The model's statements were not sufficiently supported by the "
                        "cited passages, so the answer was withheld."
                    )
                    suggestion = (
                        "Try rephrasing, or switch to keyword search to inspect exact "
                        "excerpts yourself."
                    )
                else:
                    status = "unsupported_question"
                    missing_information = getattr(draft, "missing_information", "") or ""
                    if not grounded and usage.get("record_gate_withheld"):
                        reason = (
                            "The evidence check did not find the requested individual record."
                        )
                    elif not grounded and usage.get("subject_scope_withheld"):
                        reason = (
                            "The retrieved sources did not match the named subject or "
                            "record. Try source search."
                        )
                    elif not grounded and usage.get("model_called") is False:
                        reason = (
                            "No source excerpts fit the answer length limits. Try source search."
                        )
                    else:
                        reason = missing_information or (
                            "The document does not appear to contain the information "
                            "needed to answer this question."
                        )
                    suggestion = (
                        "Check that the right document is selected, or open the original "
                        "PDF; pages with extraction warnings may not be fully indexed."
                    )
            except GenerationError as exc:
                if exc.code in {"INVALID_CITATION", "MODEL_INVALID_OUTPUT"}:
                    status = "unsupported_question"
                    reason = (
                        "The model's answer failed source validation and was withheld."
                    )
                    suggestion = "Try rephrasing the question or asking again."
                elif exc.code == "MODEL_TRUNCATED":
                    status = "unsupported_question"
                    reason = (
                        "The model's answer ran over its output limit twice and was "
                        "withheld; nothing unverified was shown."
                    )
                    suggestion = "Try a narrower question, or ask again."
                else:
                    fail(exc.status_code, exc.code, exc.message)
            finally:
                generation_ms = (time.perf_counter() - generation_started) * 1000
                generation_slot.release()
        else:
            usage = {"model_called": False}
            if mode == "bm25":
                reason = (
                    "Keyword search found no passage containing these words in the "
                    "selected documents."
                )
                suggestion = (
                    "Try different words, or switch to Hybrid + reranking, which also "
                    "matches by meaning."
                )
            else:
                reason = "No indexed passage matches this question in the selected documents."
                suggestion = (
                    "The information may not be in the document. Try different words or "
                    "check the original PDF."
                )
        if status == "answered" and coverage and not coverage["complete"]:
            reason = (
                f"Partial coverage: {coverage['pages_used']} of {coverage['pages_total']} "
                "indexed pages were read. " + reason
            )
        result = {
            "id": str(uuid4()),
            "question": body.question,
            "status": status,
            "reason": reason,
            "claims": claims,
            "evidence": evidence,
            "verification": "grounded_citation_v1"
            if grounded
            else "extractive_source_binding",
            "retrieval_mode": mode,
            "retrieval_collection": (
                index.collection if index and mode not in {"bm25", "coverage"} else None
            ),
            "model": settings.ollama_model
            if evidence and usage.get("model_called") is not False
            else None,
            "timings": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "usage": {
                **usage,
                "intent": intent["kind"],
                "scope_source": scope_source,
                "retrieval_expanded_from_conversation": usage_expanded,
            },
            "scope": {"document_ids": scope, "source": scope_source},
            "intent": intent["kind"],
        }
        if missing_information:
            result["missing_information"] = missing_information
        if suggestion:
            result["suggestion"] = suggestion
        if coverage:
            result["coverage"] = coverage
        persist(result)
        return result

    @app.get("/api/v1/history")
    def history():
        runs = []
        with connection(settings.database) as con:
            rows = con.execute(
                "SELECT created_at, response FROM query_runs ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            for row in rows:
                runs.append((row["created_at"], json.loads(row["response"])))
            for row in con.execute(
                "SELECT created_at, response FROM assistant_runs ORDER BY created_at DESC LIMIT 20"
            ).fetchall():
                runs.append((row["created_at"], json.loads(row["response"])))
        runs.sort(key=lambda item: -item[0])
        return [response for _created_at, response in runs[:20]]

    if settings.serve_frontend:
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return app


app = create_app()
