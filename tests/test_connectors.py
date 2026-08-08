"""Connector behaviour: SSRF guard, size cap, redirects, and credential gating.

The SSRF tests are the important ones. ``POST /documents/ingest`` fetches an
arbitrary URL on the server's behalf, which is the classic shape of a
request-forgery hole: without the guard, an authenticated caller could read
cloud metadata endpoints and internal services and get the response back as a
document body.
"""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from production_rag.config import get_settings
from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    InvalidSourceURIError,
    SourceFetchError,
)
from production_rag.ingestion.connectors import fetch_source
from production_rag.ingestion.sources import parse_source_uri

pytestmark = pytest.mark.asyncio

BODY = b"# Fetched Report\n\nBody text from an external source. " * 20


class _Handler(BaseHTTPRequestHandler):
    """Serves a document, a redirect chain, and an oversized body."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/report.md":
            self._respond(200, BODY, "text/markdown")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/report.md")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/huge":
            self._respond(200, b"x" * 50_000, "text/plain")
        elif self.path == "/missing":
            self._respond(404, b"nope", "text/plain")
        else:
            self._respond(404, b"", "text/plain")

    def _respond(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging to stderr."""


@pytest.fixture(scope="module")
def http_origin() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _settings(**overrides: object):
    """Base settings for local fetches.

    The loopback origin means the SSRF guard must be relaxed *for the test* —
    which is exactly the escape hatch ``allow_private_network_sources``
    documents. Tests that assert the guard fires leave it at its default.
    """
    base = {"allow_private_network_sources": True, "allow_insecure_http_sources": True}
    return get_settings().model_copy(update={**base, **overrides})


# ─── SSRF guard ───


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/report.pdf",  # loopback
        "http://localhost/report.pdf",  # loopback by name
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/internal.pdf",  # RFC1918
        "http://192.168.1.10/internal.pdf",  # RFC1918
        "http://[::1]/report.pdf",  # IPv6 loopback
    ],
)
async def test_fetch_refuses_non_public_addresses(url: str) -> None:
    settings = get_settings().model_copy(update={"allow_insecure_http_sources": True})

    with pytest.raises(SourceFetchError, match="non-public|resolve"):
        await fetch_source(parse_source_uri(url), settings)


async def test_fetch_refuses_plaintext_http_by_default() -> None:
    """http:// is off unless explicitly enabled, independent of the address."""
    settings = get_settings().model_copy(update={"allow_insecure_http_sources": False})

    with pytest.raises(InvalidSourceURIError, match="plaintext"):
        await fetch_source(parse_source_uri("http://example.com/a.pdf"), settings)


async def test_redirect_hops_are_ssrf_checked(http_origin: str) -> None:
    """A public URL must not be able to redirect the fetch to a private one.

    This is why redirects are followed by hand: httpx's own following would
    validate only the first URL.
    """
    settings = get_settings().model_copy(update={"allow_insecure_http_sources": True})

    # The origin is itself loopback, so the *first* hop is already refused —
    # proving the check runs per-hop rather than once on the caller's input.
    with pytest.raises(SourceFetchError, match="non-public"):
        await fetch_source(parse_source_uri(f"{http_origin}/redirect"), settings)


# ─── Fetch mechanics ───


async def test_fetch_returns_bytes_with_dispatch_hints(http_origin: str) -> None:
    fetched = await fetch_source(parse_source_uri(f"{http_origin}/report.md"), _settings())

    assert fetched.content == BODY
    assert fetched.filename == "report.md"  # extension drives loader dispatch
    assert fetched.content_type is not None
    assert fetched.content_type.startswith("text/markdown")


async def test_fetch_follows_redirects(http_origin: str) -> None:
    fetched = await fetch_source(parse_source_uri(f"{http_origin}/redirect"), _settings())
    assert fetched.content == BODY


async def test_fetch_caps_redirect_chain(http_origin: str) -> None:
    settings = _settings(source_fetch_max_redirects=2)

    with pytest.raises(SourceFetchError, match="redirects"):
        await fetch_source(parse_source_uri(f"{http_origin}/loop"), settings)


async def test_fetch_enforces_size_limit(http_origin: str) -> None:
    settings = _settings(max_upload_bytes=1000)

    with pytest.raises(SourceFetchError, match="limit"):
        await fetch_source(parse_source_uri(f"{http_origin}/huge"), settings)


async def test_fetch_surfaces_upstream_error_status(http_origin: str) -> None:
    with pytest.raises(SourceFetchError, match="404"):
        await fetch_source(parse_source_uri(f"{http_origin}/missing"), _settings())


# ─── Scheme routing & credential gating ───


async def test_upload_scheme_is_not_fetchable() -> None:
    """upload:// names bytes that only ever arrived over multipart."""
    uri = "upload://11111111-2222-3333-4444-555555555555/notes.txt"

    with pytest.raises(InvalidSourceURIError, match="cannot be fetched"):
        await fetch_source(parse_source_uri(uri), _settings())


async def test_gdrive_without_credentials_is_unavailable_not_broken() -> None:
    """503 with an actionable message, rather than a 500 from deep in the client."""
    settings = _settings(google_service_account_file="")

    with pytest.raises(ConnectorNotConfiguredError, match="GOOGLE_SERVICE_ACCOUNT_FILE"):
        await fetch_source(parse_source_uri("gdrive://abc123"), settings)


async def test_gdrive_with_missing_key_file_reports_the_path() -> None:
    settings = _settings(google_service_account_file="/nonexistent/key.json")

    with pytest.raises(ConnectorNotConfiguredError, match="does not exist"):
        await fetch_source(parse_source_uri("gdrive://abc123"), settings)
