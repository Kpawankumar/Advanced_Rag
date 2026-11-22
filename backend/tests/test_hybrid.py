import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient
from omnirag.db import SCHEMA, connection, initialize
from omnirag.encoders import token_windows
from omnirag.hybrid import HybridIndex, RetrievalError, reciprocal_rank_fusion
from omnirag.main import create_app
from omnirag.worker import run_once
from tokenizers import Tokenizer, models, pre_tokenizers

from .conftest import TestGateway, make_pdf
from .test_workflow import upload


class FakeEncoders:
    """Deterministic vectors for lifecycle tests, never used in production or live evaluation."""

    signature = "test-vectors"
    dimensions = 2

    def windows(self, text):
        return [text]

    def embed(self, texts):
        return [[1.0, 0.0] if "delivery" in t.lower() else [0.0, 1.0] for t in texts]

    def query(self, question):
        return self.embed([question])[0]

    def rerank(self, question, texts):
        return [1.0 if question.lower() in t.lower() else 0.0 for t in texts]


def test_v1_migration_preserves_existing_data_and_is_idempotent(tmp_path):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.execute("PRAGMA user_version=1")
        con.execute(
            "INSERT INTO documents(id,name,sha256,size_bytes,created_at,status) "
            "VALUES('old','old.pdf','hash',1,1,'ready')"
        )
        con.execute("INSERT INTO query_runs VALUES('q',1,'previous','{}')")
    initialize(path)
    initialize(path)
    with connection(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 5
        assert con.execute("SELECT question FROM query_runs").fetchone()[0] == "previous"
        assert con.execute("SELECT status FROM vector_jobs").fetchone()[0] == "pending"


def test_unknown_schema_is_not_downgraded(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA user_version=6")
    with pytest.raises(RuntimeError):
        initialize(path)


def test_rrf_uses_ranks_not_score_scales_and_deduplicates():
    a = {"chunk_id": "a", "score": -10000}
    b = {"chunk_id": "b", "score": 0.01}
    c = {"chunk_id": "c", "score": 0.99}
    result = reciprocal_rank_fusion([[a, a, b], [b, c]])
    assert [r["chunk_id"] for r in result] == ["b", "a", "c"]
    assert result[1]["score"] == pytest.approx(1 / 61)


def test_token_windows_preserve_tail_and_source_offsets():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    text = " ".join(f"word{i}" for i in range(600))
    windows = token_windows(text, tokenizer)
    assert len(windows) == 3
    assert windows[-1].endswith("word599")
    assert all(window in text for window in windows)
    assert all(len(tokenizer.encode(window).ids) <= 256 for window in windows)
    assert "word224" in windows[0] and windows[1].startswith("word224")


@pytest.fixture
def index(client, settings, sample_pdf):
    upload(client, sample_pdf)
    run_once(settings)
    index = HybridIndex(settings, FakeEncoders())
    index.open()
    yield index
    index.close()


def test_pending_gate_scope_and_restart(index, client, settings):
    with pytest.raises(RetrievalError, match="Semantic indexing"):
        index.search("delivery", [], "hybrid", 4)
    assert index.sync_one()
    first = index.search("delivery", [], "dense", 4)[0]
    assert first["page"] == 1
    second = upload(client, make_pdf(["Warranty lasts exactly 77 months."]))
    run_once(settings)
    assert index.sync_one()
    scoped = index.search("delivery", [second], "hybrid_rerank", 4)
    assert {row["document_id"] for row in scoped} == {second}
    before = index.search("delivery", [], "hybrid", 4)
    index.close()
    reopened = HybridIndex(settings, FakeEncoders())
    reopened.open()
    try:
        assert not reopened.sync_one()
        assert reopened.search("delivery", [], "hybrid", 4) == before
    finally:
        reopened.close()


def test_interrupted_job_replays_without_duplicate_points(index, settings):
    index.sync_one()
    count = index.client.count(index.collection).count
    with connection(settings.database) as con:
        con.execute("UPDATE vector_jobs SET status='processing'")
    index.close()
    reopened = HybridIndex(settings, FakeEncoders())
    reopened.open()
    try:
        assert reopened.sync_one()
        assert reopened.client.count(reopened.collection).count == count
    finally:
        reopened.close()


def test_changed_encoder_creates_new_collection_and_reindexes(index, settings):
    index.sync_one()
    old = index.collection
    index.close()
    encoder = FakeEncoders()
    encoder.signature = "updated-model"
    replacement = HybridIndex(settings, encoder)
    replacement.open()
    try:
        assert replacement.collection != old
        with pytest.raises(RetrievalError):
            replacement.search("delivery", [], "dense", 4)
        assert replacement.sync_one()
        assert replacement.search("delivery", [], "dense", 4)
    finally:
        replacement.close()


def test_failed_write_is_visible_and_retry_is_idempotent(index, settings, monkeypatch):
    original = index._upsert
    monkeypatch.setattr(index, "_upsert", lambda _batch: (_ for _ in ()).throw(OSError("disk")))
    index.sync_one()
    assert index.status()["documents"] == {"failed": 1}
    assert not index.sync_one()
    with pytest.raises(RetrievalError):
        index.search("delivery", [], "hybrid", 4)
    monkeypatch.setattr(index, "_upsert", original)
    with connection(settings.database) as con:
        con.execute("UPDATE vector_jobs SET status='pending'")
    assert index.sync_one()
    assert index.client.count(index.collection).count == 2


def test_stale_payload_is_not_used_as_citation(index, settings):
    index.sync_one()
    with connection(settings.database) as con:
        con.execute("UPDATE chunks SET text='changed text'")
    assert index.search("delivery", [], "dense", 4) == []


def test_busy_search_fails_explicitly(index):
    index.sync_one()
    with index.lock, pytest.raises(RetrievalError) as exc:
        index.search("delivery", [], "dense", 4)
    assert exc.value.code == "RETRIEVAL_BUSY"


def test_search_endpoint_does_not_generate_or_silently_fallback(client, settings, sample_pdf):
    document_id = upload(client, sample_pdf)
    run_once(settings)
    result = client.post("/api/v1/search", json={"question": "delivery", "retrieval_mode": "bm25"})
    assert result.status_code == 200 and result.json()["evidence"][0]["page"] == 1
    assert client.get("/api/v1/history").json() == []
    result = client.post("/api/v1/search", json={"question": "delivery", "retrieval_mode": "dense"})
    assert result.status_code == 503 and result.json()["detail"]["code"] == "RETRIEVAL_DISABLED"
    assert (
        client.post(
            "/api/v1/search", json={"question": "delivery", "retrieval_mode": "invented"}
        ).status_code
        == 422
    )
    assert client.get(f"/api/v1/documents/{document_id}/file").content == sample_pdf


def test_hybrid_answer_uses_unchanged_provenance_and_persists_mode(index, settings):
    index.sync_one()
    # The already-open real Qdrant index is injected without another background owner.
    index.start = lambda: None
    with TestClient(create_app(settings, TestGateway(), index=index)) as client:
        response = client.post(
            "/api/v1/queries",
            json={
                "question": "delivery",
                "retrieval_mode": "hybrid_rerank",
            },
        )
        assert response.status_code == 200
        answer = response.json()
        assert answer["retrieval_mode"] == "hybrid_rerank"
        citation = answer["claims"][0]["citations"][0]
        assert citation["page"] == 1 and "14 calendar days" in citation["quote"]
        assert client.get("/api/v1/history").json()[0] == answer


def test_model_checksum_failure_before_inference(settings, tmp_path):
    import json

    from omnirag.encoders import LocalEncoders

    settings.retrieval_models_dir = tmp_path
    settings.retrieval_manifest = tmp_path / "manifest.json"
    (tmp_path / "embedding").mkdir()
    (tmp_path / "embedding/model.onnx").write_bytes(b"bad")
    settings.retrieval_manifest.write_text(
        json.dumps(
            {
                "models": {
                    "embedding": {
                        "files": {"model.onnx": hashlib.sha256(b"good").hexdigest()},
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        LocalEncoders(settings)
