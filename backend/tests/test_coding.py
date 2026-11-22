"""Coding assistant, general chat and text/URL ingestion tests.

The coding gateway is a deterministic test double: no model server is needed and
the tests verify routing, context assembly, grounding, history and deletion.
"""

import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient
from omnirag.coding import (
    OpenRouterTextGateway,
    detect_language,
    detect_task,
)
from omnirag.config import Settings
from omnirag.db import connection, initialize
from omnirag.main import create_app


class TextGatewayStub:
    """Records requests and returns a deterministic fenced-code answer."""

    name = "stub"

    def __init__(self):
        self.calls = []
        self.answer = (
            "Here you go.\n```python\ndef is_prime(n: int) -> bool:\n"
            "    return True\n```\n"
        )

    def ready(self):
        return {"reachable": True, "model_available": True, "model": "stub-code"}

    def generate_text(self, system, messages, max_tokens, deadline):
        self.calls.append(
            {"system": system, "messages": messages, "max_tokens": max_tokens}
        )
        return self.answer, {"input_tokens": 10, "output_tokens": 20, "model_called": True}


@pytest.fixture
def code_gateway():
    return TextGatewayStub()


@pytest.fixture
def code_client(settings, code_gateway):
    with TestClient(create_app(settings, TestGatewayLike(), coding_gateway=code_gateway)) as client:
        yield client


class TestGatewayLike:
    """Minimal grounded-path stand-in so /api/v1/queries stays functional."""

    __test__ = False

    def ready(self):
        return {"reachable": True, "model_available": True, "model": "test-double"}

    def generate_grounded(self, question, evidence, conversation=None, intent=None):
        from omnirag.generation import (
            GroundedAnswerDraft,
            GroundedCitationDraft,
            GroundedClaimDraft,
        )

        text = evidence[0]["text"][:200]
        return GroundedAnswerDraft(
            answerable=True,
            claims=[
                GroundedClaimDraft(
                    text=text,
                    citations=[GroundedCitationDraft(chunk_id=evidence[0]["chunk_id"], quote=text)],
                )
            ],
        ), {"input_tokens": 100, "output_tokens": 20, "model_called": True}


# --- Task and language detection -------------------------------------------------


@pytest.mark.parametrize(
    "question,task",
    [
        ("Write a Python function that checks whether a number is prime.", "generate"),
        ("Create a React component for a dropdown.", "generate"),
        ("Why does this raise IndexError in my loop?", "debug"),
        ("Explain this function line by line.", "explain"),
        ("Refactor this for better performance.", "improve"),
        ("Review my code for security issues.", "review"),
        ("Optimize this slow query.", "improve"),
        ("Fix the traceback in my script.", "debug"),
    ],
)
def test_detect_task_routes_common_phrasings(question, task):
    assert detect_task(question) == task


def test_detect_language_from_code_and_question():
    assert detect_language("def f():\n    return 1") == "python"
    assert detect_language("", "write a python function") == "python"
    assert detect_language("const x = () => console.log(x)") == "javascript"
    assert detect_language("SELECT * FROM users WHERE id = 1") == "sql"
    assert detect_language("just text", "nothing specific") == ""


# --- Coding assistant endpoint ----------------------------------------------------


def test_coding_generate_returns_fenced_answer(code_client, code_gateway):
    response = code_client.post(
        "/api/v1/coding/assist",
        json={"question": "Write a Python function that checks whether a number is prime."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["kind"] == "coding"
    assert body["task"] == "generate"
    assert "```python" in body["answer"]
    assert body["language"] == "python"
    assert body["model"] is not None
    assert code_gateway.calls, "the coding gateway was never called"
    request = json.loads(code_gateway.calls[0]["messages"][0]["content"])
    assert request["request"].startswith("Write a Python function")


def test_coding_debug_includes_code_and_error(code_client, code_gateway):
    response = code_client.post(
        "/api/v1/coding/assist",
        json={
            "question": "Why does this crash?",
            "task": "debug",
            "code": "vals = [1,2,3]\nprint(vals[10])",
            "error": "IndexError: list index out of range",
        },
    )
    assert response.status_code == 200
    request = json.loads(code_gateway.calls[-1]["messages"][0]["content"])
    assert request["code"].startswith("vals = [1,2,3]")
    assert request["error_or_output"] == "IndexError: list index out of range"


def test_coding_conversation_reaches_the_gateway(code_client, code_gateway):
    code_client.post(
        "/api/v1/coding/assist",
        json={
            "question": "Now make it handle negatives.",
            "conversation": [
                {"question": "Write a prime checker.",
                    "answer": "```python\ndef is_prime(n): ...```"}
            ],
        },
    )
    request = json.loads(code_gateway.calls[-1]["messages"][0]["content"])
    assert request["earlier_conversation"][0]["question"] == "Write a prime checker."


def test_coding_history_is_persisted_and_merged(code_client):
    code_client.post("/api/v1/coding/assist", json={"question": "Write a prime checker."})
    code_client.post("/api/v1/assistant/chat", json={"question": "What is a heap?"})
    history = code_client.get("/api/v1/history").json()
    kinds = [item["kind"] for item in history]
    assert "coding" in kinds and "chat" in kinds
    coding = next(item for item in history if item["kind"] == "coding")
    assert coding["answer"].startswith("Here you go.")


def test_empty_question_is_rejected(code_client):
    response = code_client.post("/api/v1/coding/assist", json={"question": ""})
    assert response.status_code == 422


# --- Knowledge-base-aware coding --------------------------------------------------


def test_coding_with_document_scope_grounds_and_lists_sources(code_client, code_gateway, settings,
    sample_pdf):
    upload = code_client.post("/api/v1/documents", files={"file": ("doc.pdf", sample_pdf)})
    document_id = upload.json()["document"]["id"]
    # The worker normally indexes the PDF; emulate a ready text pipeline result.
    with connection(settings.database) as con:
        con.execute(
            "UPDATE documents SET status='ready', page_count=1 WHERE id=?", (document_id,)
        )
        con.execute(
            "INSERT INTO page_extractions VALUES(?,?,?,?)",
            (document_id, 1, "native", json.dumps({"page": 1, "method": "native"})),
        )
        con.execute(
            "INSERT INTO chunks VALUES(?,?,?,?,?)",
            (f"{document_id}:1:0", document_id, 1, 0, "Orion Supplies delivery window is 14 days."),
        )
        con.execute(
            "INSERT INTO chunks_fts VALUES(?,?)",
            (f"{document_id}:1:0", "Orion Supplies delivery window is 14 days."),
        )
    response = code_client.post(
        "/api/v1/coding/assist",
        json={
            "question": "Using the uploaded documentation, write a delivery-window constant.",
            "document_ids": [document_id],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"], "grounded coding must return the passages used"
    assert body["sources"][0]["document_id"] == document_id
    request = json.loads(code_gateway.calls[-1]["messages"][0]["content"])
    assert request["project_documentation"][0]["text"].startswith("Orion Supplies")


def test_coding_grounding_prompt_forbids_inventing_details():
    from omnirag.coding import build_system_prompt

    system = build_system_prompt("generate", grounded=True)
    assert "do not invent project details" in system
    assert "[source_id]" in system
    # Un-grounded requests keep the general task rules without the doc addendum.
    assert "do not invent project details" not in build_system_prompt("generate", grounded=False)


# --- General chat -----------------------------------------------------------------


def test_general_chat_is_not_grounded(code_client, code_gateway):
    response = code_client.post(
        "/api/v1/assistant/chat", json={"question": "What does SOLID stand for?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "chat"
    assert body["task"] == "chat"
    assert body["grounded"] is False
    assert body["sources"] == []
    assert "coding assistant" in code_gateway.calls[-1]["system"].lower()


# --- Text format ingestion ---------------------------------------------------------


def test_markdown_upload_is_indexed_immediately(settings):
    gateway = TextGatewayStub()
    with TestClient(create_app(settings, TestGatewayLike(), coding_gateway=gateway)) as client:
        content = (
            b"# Project API\n\nThe service exposes `POST /deploy` which runs a build. "
            b"Authentication uses a bearer token issued by the admin console. "
            b"Rate limits are ten requests per minute."
        )
        response = client.post("/api/v1/documents", files={"file": ("api.md", content)})
        assert response.status_code == 202
        document = response.json()["document"]
        assert document["status"] == "ready"
        assert document["chunk_count"] >= 1
        answer = client.post(
            "/api/v1/search",
            json={"question": "deploy endpoint authentication"},
        )
        assert any(
            hit["document_id"] == document["id"] for hit in answer.json()["evidence"]
        )


def test_empty_text_upload_fails_visibly(settings):
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        response = client.post("/api/v1/documents", files={"file": ("empty.txt", b"")})
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EMPTY_DOCUMENT"


def test_json_upload_is_rejected_when_invalid(settings):
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        response = client.post("/api/v1/documents", files={"file": ("broken.json", b"{not json")})
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "INVALID_JSON"


def test_unsupported_extension_is_rejected(settings):
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        response = client.post("/api/v1/documents", files={"file": ("virus.exe", b"MZ")})
        assert response.status_code == 415


# --- URL ingestion (SSRF guards) ----------------------------------------------------


def test_url_to_private_host_is_blocked(settings):
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        for target in ("http://127.0.0.1:8000/", "http://localhost/x", "http://192.168.1.4/a"):
            response = client.post("/api/v1/urls", json={"url": target})
            assert response.status_code == 422
        response = client.post("/api/v1/urls", json={"url": "ftp://example.com/x"})
        assert response.status_code == 422


def test_url_ingestion_stores_ready_document(settings, monkeypatch):
    from omnirag import ingestion as omnirag_ingestion

    html = (
        "<html><head><title>Deploy Guide</title></head><body>"
        "<script>evil()</script><p>The deploy endpoint accepts a bearer token.</p>"
        "<p>Rate limits are ten requests per minute for every account.</p>"
        "</body></html>"
    )

    class FakeResponse:
        status_code = 200
        content = html.encode()
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://docs.example.com/deploy"
        text = html
        is_redirect = False

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(omnirag_ingestion.httpx, "Client", FakeClient)
    # The sandbox cannot resolve example.com; the guard's DNS step is stubbed so the
    # unit under test stays the SSRF policy and the ingestion pipeline.
    import socket as _socket

    monkeypatch.setattr(
        omnirag_ingestion.socket,
        "getaddrinfo",
        lambda host, port: [(_socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        response = client.post("/api/v1/urls", json={"url": "https://docs.example.com/deploy"})
        assert response.status_code == 202
        document = response.json()["document"]
        assert document["status"] == "ready"
        assert document["name"] == "Deploy Guide"
        assert document["source_url"] == "https://docs.example.com/deploy"
        search = client.post(
            "/api/v1/search", json={"question": "bearer token deploy rate limits"}
        ).json()
        assert any(hit["document_id"] == document["id"] for hit in search["evidence"])


# --- Document deletion --------------------------------------------------------------


def test_delete_removes_document_and_derived_data(settings, sample_pdf):
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        upload = client.post("/api/v1/documents", files={"file": ("doc.pdf", sample_pdf)})
        document_id = upload.json()["document"]["id"]
        deleted = client.delete(f"/api/v1/documents/{document_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "id": document_id}
        assert client.get(f"/api/v1/documents/{document_id}").status_code == 404
        with connection(settings.database) as con:
            assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
            assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
        assert not list(settings.uploads.glob("*.pdf"))
        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 404


def test_delete_removes_text_document(settings):
    content = ("# Notes\n\n" + "Remember the deploy token rotation rule. " * 20).encode()
    with TestClient(create_app(settings, TestGatewayLike(),
        coding_gateway=TextGatewayStub())) as client:
        upload = client.post("/api/v1/documents", files={"file": ("notes.md", content)})
        document_id = upload.json()["document"]["id"]
        assert client.delete(f"/api/v1/documents/{document_id}").json()["deleted"] is True
        with connection(settings.database) as con:
            assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


# --- Readiness ----------------------------------------------------------------------


def test_readiness_reports_coding_provider(settings, code_gateway):
    with TestClient(create_app(settings, TestGatewayLike(), coding_gateway=code_gateway)) as client:
        readiness = client.get("/api/v1/readiness").json()
        assert readiness["coding"]["provider"] == "ollama"
        assert readiness["coding"]["model_available"] is True
        assert readiness["coding"]["model"] == "stub-code"


# --- OpenRouter gateway (ported from the RAG project) --------------------------------


def test_openrouter_gateway_requires_api_key():
    settings = Settings(_env_file=None, code_provider="openrouter", openrouter_api_key="")
    gateway = OpenRouterTextGateway(settings)
    assert gateway.ready()["reachable"] is False
    with pytest.raises(Exception) as excinfo:
        gateway.generate_text("sys", [{"role": "user", "content": "hi"}], 100, 1e18)
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_openrouter_gateway_sends_ranking_headers_and_payload():
    settings = Settings(
        _env_file=None,
        code_provider="openrouter",
        openrouter_api_key="sk-or-test",
        openrouter_model="openai/gpt-5.2",
        openrouter_http_referer="http://127.0.0.1:8000",
        openrouter_title="OmniRAG AI",
    )
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer text"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        )

    transport = httpx.MockTransport(handler)
    gateway = OpenRouterTextGateway(settings, transport)
    text, usage = gateway.generate_text(
        "system prompt", [{"role": "user", "content": "hello"}], 256, 1e18
    )
    assert text == "answer text"
    assert usage["output_tokens"] == 7
    assert seen["headers"].get("http-referer") == "http://127.0.0.1:8000"
    assert seen["headers"].get("x-openrouter-title") == "OmniRAG AI"
    assert seen["payload"]["model"] == "openai/gpt-5.2"
    assert seen["payload"]["messages"][0]["role"] == "system"


def test_openrouter_rate_limit_maps_to_429():
    settings = Settings(_env_file=None, code_provider="openrouter", openrouter_api_key="k")

    def handler(request):
        return httpx.Response(429, json={"error": "quota"})

    gateway = OpenRouterTextGateway(settings, httpx.MockTransport(handler))
    with pytest.raises(Exception) as excinfo:
        gateway.generate_text("s", [{"role": "user", "content": "q"}], 10, 1e18)
    assert getattr(excinfo.value, "status_code", None) == 429


# --- Migration 5 ---------------------------------------------------------------------


def test_migration_5_creates_assistant_runs_and_source_url(tmp_path):
    path = tmp_path / "v4.db"
    from omnirag.db import SCHEMA as BASE_SCHEMA

    with sqlite3.connect(path) as con:
        con.executescript(BASE_SCHEMA)
        con.execute("PRAGMA user_version=4")
    initialize(path)
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 5
        columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        assert "source_url" in columns
        con.execute("INSERT INTO assistant_runs VALUES('r',1,'coding','generate','q','{}')")
