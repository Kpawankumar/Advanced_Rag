import os
import time

import pytest
from fastapi.testclient import TestClient
from omnirag.auth import COOKIE, AuthStore
from omnirag.main import create_app

PASSWORD = "a long test password only"


@pytest.fixture
def owner(settings):
    settings.auth_enabled = True
    store = AuthStore(settings.data_dir / "auth")
    store.initialize()
    store.set_password(PASSWORD)
    return store


def login(client, password=PASSWORD):
    return client.post(
        "/api/v1/auth/login",
        json={"password": password},
        headers={"X-OmniRAG-Login": "1", "Origin": "http://127.0.0.1:8000"},
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/documents",
        "/api/v1/history",
        "/api/v1/readiness",
        "/api/v1/documents/00000000-0000-0000-0000-000000000001/file",
        "/docs",
        "/openapi.json",
    ],
)
def test_anonymous_cannot_read_private_routes(settings, owner, path):
    with TestClient(create_app(settings)) as client:
        assert client.get(path).status_code == 401
        assert client.post("/api/v1/queries", json={"question": "anything"}).status_code == 401
        assert client.get("/api/v1/health").json()["authentication"] is True


def test_login_csrf_logout_and_cache(settings, owner):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/auth/session").json()["authenticated"] is False
        result = login(client)
        assert result.status_code == 200
        cookie = result.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert result.headers["cache-control"] == "no-store"
        csrf = result.json()["csrf"]
        token = client.cookies.get(COOKIE)
        with owner.connect() as db:
            assert token not in str([tuple(r) for r in db.execute("SELECT * FROM sessions")])
        assert client.get("/api/v1/documents").status_code == 200
        assert client.get("/api/v1/history").headers["cache-control"] == "no-store"
        assert client.post("/api/v1/auth/logout").status_code == 403
        assert (
            client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
        )
        assert (
            client.post(
                "/api/v1/auth/logout",
                headers={"X-CSRF-Token": csrf, "Origin": "https://hostile.example"},
            ).status_code
            == 403
        )
        assert client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
        client.cookies.set(COOKIE, token)
        assert client.get("/api/v1/history").status_code == 401


def test_login_requires_custom_header_and_valid_origin(settings, owner):
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/auth/login", json={"password": PASSWORD}).status_code == 403
        assert (
            client.post(
                "/api/v1/auth/login",
                json={"password": PASSWORD},
                headers={"X-OmniRAG-Login": "1", "Origin": "https://hostile.example"},
            ).status_code
            == 403
        )
        assert login(client, "incorrect").status_code == 401


def test_rate_limit_persists_across_app_restart(settings, owner):
    with TestClient(create_app(settings)) as client:
        for _ in range(5):
            assert login(client, "incorrect").status_code == 401
    with TestClient(create_app(settings)) as client:
        assert login(client).status_code == 429


@pytest.mark.parametrize("column,value", [("expires", 0), ("touched", time.time() - 1801)])
def test_session_expires(settings, owner, column, value):
    with TestClient(create_app(settings)) as client:
        assert login(client).status_code == 200
        with owner.connect() as db:
            db.execute(f"UPDATE sessions SET {column}=?", (value,))
        assert client.get("/api/v1/history").status_code == 401


def test_password_change_revokes_sessions_and_survives_restart(settings, owner):
    with TestClient(create_app(settings)) as client:
        result = login(client)
        token = client.cookies.get(COOKIE)
        csrf = result.json()["csrf"]
        response = client.post(
            "/api/v1/auth/password",
            json={"current_password": PASSWORD, "new_password": "another long test password"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        client.cookies.set(COOKIE, token)
        assert client.get("/api/v1/history").status_code == 401
    with TestClient(create_app(settings)) as client:
        assert login(client).status_code == 401
        assert login(client, "another long test password").status_code == 200


def test_session_survives_restart_and_new_login_rotates_it(settings, owner):
    with TestClient(create_app(settings)) as client:
        login(client)
        token = client.cookies.get(COOKIE)
    with TestClient(create_app(settings)) as client:
        client.cookies.set(COOKIE, token)
        assert client.get("/api/v1/history").status_code == 200
        login(client)
        assert owner.session(token) is None


def test_credential_permissions_and_no_overwrite(owner):
    # POSIX permission bits only exist on POSIX; Windows reports a stub mask.
    if os.name == "posix":
        assert os.stat(owner.path).st_mode & 0o777 == 0o600
        assert os.stat(owner.directory).st_mode & 0o777 == 0o700
    with pytest.raises(ValueError):
        owner.set_password("a different long password")
    assert owner.verify(PASSWORD)


def test_unconfigured_auth_fails_closed(settings):
    settings.auth_enabled = True
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/auth/session").json()["configured"] is False
        assert login(client).status_code == 401
        assert client.get("/api/v1/documents").status_code == 401


def test_built_frontend_is_served_without_exposing_documents(settings, owner, tmp_path):
    settings.frontend_dist = tmp_path / "web"
    settings.frontend_dist.mkdir()
    (settings.frontend_dist / "index.html").write_text("<html>Sign in</html>")
    settings.serve_frontend = True
    with TestClient(create_app(settings)) as client:
        assert "Sign in" in client.get("/").text
        assert client.get("/api/v1/documents").status_code == 401
        assert client.get("/../omnirag.sqlite3").status_code == 404
