import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

from omnirag.config import Settings
from omnirag.db import (
    claim_job,
    connection,
    document_chunks,
    finish_job,
    get_summary,
    initialize,
    save_summary,
)

log = logging.getLogger("omnirag.worker")


def run_parser(command, timeout):
    # The process group includes Tesseract; killing only Python leaves OCR orphaned.
    with TemporaryDirectory(prefix="omnirag-parser-") as directory, subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "TMPDIR": directory, "TMP": directory, "TEMP": directory},
    ) as proc:
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except BaseException:
            try:
                if hasattr(os, "killpg"):
                    # The process group includes Tesseract; killing only Python
                    # leaves OCR orphaned.
                    os.killpg(proc.pid, signal.SIGKILL)
                elif os.name == "nt":
                    # Windows has no process groups; taskkill walks the process tree.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.communicate()
            raise
        if proc.returncode:
            raise ValueError("Parser process failed")
        return json.loads(stdout)


def generate_overview(settings: Settings, document_id: str, name: str) -> None:
    """Best-effort overview right after ingestion; never fails the document.

    Existing documents are not backfilled at startup: generation contends with
    owner queries for the single local model slot, so older documents get their
    overview on demand from the workspace.
    """
    if get_summary(settings.database, document_id, "overview") is not None:
        return
    from omnirag.generation import GenerationError, OllamaGateway
    from omnirag.summary import generate_overview

    try:
        chunks = document_chunks(settings.database, document_id)
        if not chunks:
            return
        gateway = OllamaGateway(settings)
        content, coverage = generate_overview(gateway, name, chunks)
        save_summary(
            settings.database,
            document_id,
            "overview",
            status="ready",
            content=content,
            coverage=coverage,
            model=settings.ollama_model,
            error=None,
        )
        log.info("overview generated document=%s", document_id)
    except GenerationError as exc:
        save_summary(
            settings.database,
            document_id,
            "overview",
            status="failed",
            content=None,
            coverage=None,
            model=None,
            error=f"{exc.code}: {exc.message}",
        )
        log.warning("overview failed document=%s code=%s", document_id, exc.code)
    except Exception:
        log.exception("Unexpected overview failure document=%s", document_id)
        save_summary(
            settings.database,
            document_id,
            "overview",
            status="failed",
            content=None,
            coverage=None,
            model=None,
            error="Unexpected error during overview generation.",
        )


def run_once(settings: Settings) -> bool:
    with connection(settings.database) as con:
        con.execute(
            "INSERT OR REPLACE INTO runtime(key,value) VALUES('worker_heartbeat',?)", (time.time(),)
        )
    job = claim_job(settings.database, settings.ingest_timeout_seconds)
    if job is None:
        return False
    try:
        result = run_parser(
            [
                sys.executable,
                "-m",
                "omnirag.ingestion",
                str(settings.uploads / f"{job['id']}.pdf"),
                str(settings.max_pages),
                json.dumps(
                    {
                        "ocr_enabled": settings.ocr_enabled,
                        "ocr_max_pages": settings.ocr_max_pages,
                        "ocr_page_timeout_seconds": settings.ocr_page_timeout_seconds,
                    }
                ),
            ],
            timeout=settings.ingest_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        result = {"error_code": "PARSER_TIMEOUT", "error_message": "PDF processing timed out."}
    except (ValueError, OSError):
        result = {"error_code": "PARSER_FAILED", "error_message": "PDF parser process failed."}
    accepted = finish_job(settings.database, job, result)
    log.info("job=%s accepted=%s result=%s", job["id"], accepted, result.get("error_code", "ready"))
    if accepted and not result.get("error_code"):
        generate_overview(settings, job["id"], job["name"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process at most one queued document.")
    args = parser.parse_args()
    settings = Settings()
    settings.prepare()
    initialize(settings.database)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        worked = run_once(settings)
        if args.once:
            break
        if not worked:
            time.sleep(1)


if __name__ == "__main__":
    main()
