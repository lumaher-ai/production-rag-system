"""Connectors — retrieve the bytes behind a fetchable source URI.

A connector's only job is to turn a ``ParsedSource`` into ``FetchedBytes``
(content + filename + content type). It deliberately knows nothing about
parsing: the returned triple is exactly what ``loaders.load_file`` already
accepts, so every format the upload path supports works over every connector
for free, and adding a format never touches this module.

Connectors are wired through a registry (scheme -> connector), mirroring
``LOADER_REGISTRY``. Adding a source type means writing one connector and
registering it — no dispatch chain to edit.

Credentials are **app-level**: they come from ``Settings``, so every user's
pull runs as the same server identity. This system therefore only reaches what
the server itself can see. Per-user credentials would require a token table,
encryption at rest, and a consent flow; that is a deliberate scope boundary,
not an oversight.
"""

import asyncio
import ipaddress
import socket
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import BaseModel

from production_rag.config import Settings
from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    InvalidSourceURIError,
    SourceFetchError,
)
from production_rag.ingestion.sources import (
    GDRIVE_SCHEME,
    HTTP_SCHEME,
    HTTPS_SCHEME,
    ParsedSource,
)
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

DRIVE_API_ROOT = "https://www.googleapis.com/drive/v3/files"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

# Google-native files have no bytes to download — they must be *exported* to a
# concrete format. HTML is chosen for Docs because HtmlLoader converts it to
# Markdown and recovers heading structure, so a Doc keeps its section
# provenance instead of collapsing into one flat blob.
GOOGLE_EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/html", ".html"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}


class FetchedBytes(BaseModel):
    """Raw bytes plus the two hints ``load_file`` dispatches on."""

    content: bytes
    filename: str
    content_type: str | None = None


class Connector(Protocol):
    async def fetch(self, parsed: ParsedSource, settings: Settings) -> FetchedBytes:
        """Retrieve the bytes a source URI points at."""
        ...


def _assert_public_address(host: str, settings: Settings) -> None:
    """Refuse to fetch from addresses that are not publicly routable.

    Server-Side Request Forgery guard. Without it, ``POST /documents/ingest``
    with ``http://169.254.169.254/...`` or ``http://localhost:5432`` would turn
    an authenticated ingest endpoint into a proxy for reaching cloud metadata
    services and other services on the host's private network — the document
    body would then hand the response back to the caller.

    Every A record is checked, not just the first, so a host that resolves to
    both a public and a private address is still rejected. This is called again
    for each redirect hop, because the guarded URL is not the URL finally
    fetched.

    Blocking is deliberately at DNS-resolution time. That leaves a narrow
    DNS-rebinding window (the name could resolve differently when httpx
    reconnects); closing it fully means pinning the resolved IP into the
    connection, which is noted here as a known limit rather than pretended away.
    """
    if settings.allow_private_network_sources:
        return
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SourceFetchError(f"Could not resolve host '{host}'.") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            logger.warning("source_fetch_blocked_address", host=host, address=str(address))
            raise SourceFetchError(
                f"Refusing to fetch from '{host}': it resolves to the non-public address {address}."
            )


async def _read_capped(response: httpx.Response, limit: int, uri: str) -> bytes:
    """Stream a response body, aborting as soon as it exceeds ``limit``.

    The Content-Length header is checked first as a cheap rejection, but the
    running total is what actually enforces the cap — a chunked response has no
    Content-Length, and a hostile origin can understate it.
    """
    declared = response.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise SourceFetchError(
            f"Source '{uri}' declares {declared} bytes, over the {limit}-byte limit."
        )

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise SourceFetchError(f"Source '{uri}' exceeds the {limit}-byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _filename_from_response(url: str, response: httpx.Response) -> str:
    """Best filename for loader dispatch: Content-Disposition, else URL basename.

    The extension matters — ``_resolve_loader`` prefers it over Content-Type
    precisely because origins so often send ``application/octet-stream``.
    """
    disposition = response.headers.get("content-disposition", "")
    if "filename=" in disposition:
        candidate = disposition.split("filename=", 1)[1].strip().strip('"; ')
        if candidate:
            return unquote(candidate)
    basename = Path(unquote(urlsplit(url).path)).name
    return basename or "document"


class HttpConnector:
    """Fetch a document over HTTP(S), following redirects by hand.

    Redirects are followed manually rather than via ``follow_redirects=True``
    so that every hop is SSRF-checked. Automatic following would validate only
    the URL the caller supplied, letting a public host redirect the fetch to
    ``169.254.169.254`` — which is the standard way this guard gets bypassed.
    """

    async def fetch(self, parsed: ParsedSource, settings: Settings) -> FetchedBytes:
        if parsed.scheme == HTTP_SCHEME and not settings.allow_insecure_http_sources:
            raise InvalidSourceURIError(
                "Refusing to fetch over plaintext http://. Use https://, or set "
                "ALLOW_INSECURE_HTTP_SOURCES=true for local development."
            )

        url = parsed.uri
        limit = settings.max_upload_bytes
        timeout = httpx.Timeout(settings.source_fetch_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            for _ in range(settings.source_fetch_max_redirects + 1):
                host = urlsplit(url).hostname
                if not host:
                    raise InvalidSourceURIError(f"Source '{url}' has no host.")
                # getaddrinfo blocks; keep it off the event loop.
                await asyncio.to_thread(_assert_public_address, host, settings)

                request = client.build_request("GET", url)
                response = await client.send(request, stream=True)
                try:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise SourceFetchError(
                                f"Source '{url}' returned {response.status_code} with no Location."
                            )
                        # Resolve relative Locations against the current URL.
                        url = str(response.next_request.url) if response.next_request else location
                        continue
                    if response.status_code >= 400:
                        raise SourceFetchError(
                            f"Source '{url}' returned HTTP {response.status_code}."
                        )
                    content = await _read_capped(response, limit, url)
                    filename = _filename_from_response(url, response)
                    content_type = response.headers.get("content-type")
                finally:
                    await response.aclose()

                logger.info(
                    "source_fetched",
                    scheme=parsed.scheme,
                    url=url,
                    bytes=len(content),
                    content_type=content_type,
                )
                return FetchedBytes(content=content, filename=filename, content_type=content_type)

        raise SourceFetchError(
            f"Source '{parsed.uri}' exceeded {settings.source_fetch_max_redirects} redirects."
        )


class GoogleDriveConnector:
    """Fetch a Drive file by id using an app-level service account.

    The service account must have been granted read access to the file (shared
    directly, or via a shared drive). Google-native types are exported rather
    than downloaded; binary types (PDF, DOCX) download as-is.
    """

    async def fetch(self, parsed: ParsedSource, settings: Settings) -> FetchedBytes:
        token = await _google_access_token(settings)
        file_id = parsed.authority
        headers = {"Authorization": f"Bearer {token}"}
        timeout = httpx.Timeout(settings.source_fetch_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            meta_response = await client.get(
                f"{DRIVE_API_ROOT}/{file_id}",
                params={"fields": "id,name,mimeType", "supportsAllDrives": "true"},
                headers=headers,
            )
            if meta_response.status_code == 404:
                raise SourceFetchError(
                    f"Drive file '{file_id}' not found, or not shared with the service account."
                )
            if meta_response.status_code >= 400:
                raise SourceFetchError(
                    f"Drive metadata lookup for '{file_id}' returned "
                    f"HTTP {meta_response.status_code}."
                )

            meta = meta_response.json()
            name = str(meta.get("name") or file_id)
            drive_mime = str(meta.get("mimeType") or "")
            filename: str
            content_type: str | None

            export = GOOGLE_EXPORT_FORMATS.get(drive_mime)
            if export is not None:
                export_mime, suffix = export
                request = client.build_request(
                    "GET",
                    f"{DRIVE_API_ROOT}/{file_id}/export",
                    params={"mimeType": export_mime},
                    headers=headers,
                )
                # Exported names carry no extension; add one so the loader
                # registry dispatches on it.
                filename = name if name.endswith(suffix) else f"{name}{suffix}"
                content_type = export_mime
            elif drive_mime.startswith("application/vnd.google-apps."):
                raise InvalidSourceURIError(
                    f"Drive file '{name}' is a {drive_mime}, which has no supported "
                    f"text export. Supported Google types: Docs, Slides."
                )
            else:
                request = client.build_request(
                    "GET",
                    f"{DRIVE_API_ROOT}/{file_id}",
                    params={"alt": "media", "supportsAllDrives": "true"},
                    headers=headers,
                )
                filename = name
                content_type = drive_mime or None

            response = await client.send(request, stream=True)
            try:
                if response.status_code >= 400:
                    raise SourceFetchError(
                        f"Drive download for '{file_id}' returned HTTP {response.status_code}."
                    )
                content = await _read_capped(response, settings.max_upload_bytes, parsed.uri)
            finally:
                await response.aclose()

        logger.info(
            "source_fetched",
            scheme=GDRIVE_SCHEME,
            file_id=file_id,
            drive_mime=drive_mime,
            bytes=len(content),
        )
        return FetchedBytes(content=content, filename=filename, content_type=content_type)


def _load_google_credentials(service_account_file: str):
    """Build read-only Drive credentials from a service-account JSON file."""
    try:
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConnectorNotConfiguredError(
            "Google Drive ingestion requires the 'google-auth' package."
        ) from exc

    return service_account.Credentials.from_service_account_file(
        service_account_file, scopes=[DRIVE_SCOPE]
    )


async def _google_access_token(settings: Settings) -> str:
    """Mint (or refresh) an access token for the configured service account.

    google-auth is synchronous and its refresh performs network I/O, so the
    whole operation runs in a worker thread. Credentials are rebuilt per call
    rather than cached — token refresh is infrequent relative to an ingest, and
    caching mutable credential state across requests is not worth the race.
    """
    path = settings.google_service_account_file
    if not path:
        raise ConnectorNotConfiguredError(
            "Google Drive ingestion is not configured. Set "
            "GOOGLE_SERVICE_ACCOUNT_FILE to a service-account JSON key path."
        )
    if not Path(path).is_file():
        raise ConnectorNotConfiguredError(
            f"GOOGLE_SERVICE_ACCOUNT_FILE points at '{path}', which does not exist."
        )

    def _refresh() -> str:
        from google.auth.transport.requests import Request as GoogleAuthRequest

        credentials = _load_google_credentials(path)
        credentials.refresh(GoogleAuthRequest())
        return str(credentials.token)

    try:
        return await asyncio.to_thread(_refresh)
    except (ConnectorNotConfiguredError, SourceFetchError):
        raise
    except Exception as exc:
        raise ConnectorNotConfiguredError(
            f"Could not obtain a Google Drive access token: {exc}"
        ) from exc


CONNECTOR_REGISTRY: dict[str, Connector] = {
    HTTPS_SCHEME: HttpConnector(),
    HTTP_SCHEME: HttpConnector(),
    GDRIVE_SCHEME: GoogleDriveConnector(),
}


async def fetch_source(parsed: ParsedSource, settings: Settings) -> FetchedBytes:
    """Dispatch a fetchable source URI to its connector.

    Raises ``InvalidSourceURIError`` for a scheme that exists but cannot be
    fetched (``upload://`` names bytes that only ever arrived over multipart).
    """
    connector = CONNECTOR_REGISTRY.get(parsed.scheme)
    if connector is None:
        raise InvalidSourceURIError(
            f"Scheme '{parsed.scheme}://' cannot be fetched. Fetchable schemes: "
            f"https://, http://, gdrive://. Files uploaded directly use "
            f"upload:// and are ingested via POST /documents/upload."
        )
    return await connector.fetch(parsed, settings)
