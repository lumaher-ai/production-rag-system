"""Source URIs — the stable identity of an ingested document.

Every document carries a ``source``, and ``(user_id, source)`` is its identity:
re-ingesting the same source replaces it in place instead of creating a second
copy. That makes ``source`` a data-model value, not a label — so it is a real
URI with a scheme, not a bare filename.

    upload://<user_id>/reporte.pdf        a file the user uploaded
    https://example.com/report.pdf        a document fetched from the web
    gdrive://<file_id>                    a file pulled from Google Drive

**Why the user id and not the user's name.** The authority segment of an
``upload://`` URI is the owner's UUID. ``User.name`` is mutable and non-unique
and ``User.email`` is mutable PII; either would change the source string when a
user is renamed, silently breaking ``(user_id, source)`` identity so that the
next upload of an unchanged file creates a duplicate rather than replacing the
original. The UUID never changes.

The scheme is what routes a URI to its connector (see ``connectors.py``), which
is why parsing lives here rather than in any one connector: adding a scheme
means registering a connector, not editing a dispatch chain.
"""

from urllib.parse import quote, unquote, urlsplit
from uuid import UUID

from pydantic import BaseModel

from production_rag.exceptions import InvalidSourceURIError

# Matches the length of ``documents.source`` (String(1024)). A URI that would
# exceed the column is rejected rather than truncated: a truncated URI is a
# *different* identity that would silently collide with, or fail to match, the
# document it names.
SOURCE_URI_MAX_LEN = 1024

UPLOAD_SCHEME = "upload"
GDRIVE_SCHEME = "gdrive"
HTTPS_SCHEME = "https"
HTTP_SCHEME = "http"

# Schemes a client may hand to the ingest-by-URI endpoint. ``upload://`` is
# absent by design — it is minted by the server when a file arrives over
# multipart and names bytes that exist nowhere else, so it can never be fetched.
FETCHABLE_SCHEMES = frozenset({HTTPS_SCHEME, HTTP_SCHEME, GDRIVE_SCHEME})

KNOWN_SCHEMES = FETCHABLE_SCHEMES | {UPLOAD_SCHEME}


class ParsedSource(BaseModel):
    """A validated source URI, split into the parts connectors dispatch on."""

    uri: str
    scheme: str
    # For upload:// the owner's UUID; for gdrive:// the file id; for http(s)
    # the host[:port].
    authority: str
    # Percent-decoded path portion, without its leading slash. Empty when the
    # scheme carries everything in the authority (e.g. gdrive://<id>).
    path: str

    @property
    def is_fetchable(self) -> bool:
        """Whether a connector can retrieve bytes for this URI."""
        return self.scheme in FETCHABLE_SCHEMES


def build_upload_uri(user_id: UUID, filename: str | None) -> str:
    """Mint the canonical ``upload://`` URI for a multipart upload.

    The filename is percent-encoded with no safe characters, so a name
    containing ``/``, ``?``, ``#`` or spaces cannot forge extra URI structure —
    ``a/b.pdf`` becomes ``a%2Fb.pdf`` and stays a single path segment.

    If the encoded URI would exceed the ``source`` column, the *filename* is
    trimmed rather than the URI, keeping the scheme and owner intact.
    """
    name = (filename or "document").strip() or "document"
    prefix = f"{UPLOAD_SCHEME}://{user_id}/"
    encoded = quote(name, safe="")
    budget = SOURCE_URI_MAX_LEN - len(prefix)
    if len(encoded) > budget:
        # Trim the raw name and re-encode: slicing an encoded string can cut a
        # percent-triplet in half and produce an undecodable URI.
        while len(encoded) > budget and name:
            name = name[:-1]
            encoded = quote(name, safe="")
    return prefix + encoded


def parse_source_uri(raw: str) -> ParsedSource:
    """Validate and split a source URI.

    Raises ``InvalidSourceURIError`` (422) if the URI is empty, over-long, has
    no scheme, uses a scheme this system does not ingest, or omits the part its
    scheme requires.
    """
    uri = (raw or "").strip()
    if not uri:
        raise InvalidSourceURIError("Source URI must not be empty.")
    if len(uri) > SOURCE_URI_MAX_LEN:
        raise InvalidSourceURIError(
            f"Source URI exceeds the maximum length of {SOURCE_URI_MAX_LEN} characters."
        )

    split = urlsplit(uri)
    scheme = split.scheme.lower()
    if not scheme:
        raise InvalidSourceURIError(
            f"Source '{uri}' is not a URI — it needs a scheme, e.g. "
            f"https://example.com/report.pdf or gdrive://<file_id>."
        )
    if scheme not in KNOWN_SCHEMES:
        raise InvalidSourceURIError(
            f"Unsupported source scheme '{scheme}://'. "
            f"Supported: {', '.join(sorted(s + '://' for s in KNOWN_SCHEMES))}."
        )

    authority = split.netloc
    if not authority:
        raise InvalidSourceURIError(
            f"Source '{uri}' is missing its authority "
            f"(expected {scheme}://<something>/...)."
        )

    path = unquote(split.path.lstrip("/"))

    if scheme in (HTTPS_SCHEME, HTTP_SCHEME) and not path:
        # Bare-host URLs are almost always a mistake (a landing page, not a
        # document), and there is no filename to dispatch a loader on.
        raise InvalidSourceURIError(
            f"Source '{uri}' points at a host with no path — give the full URL of a document."
        )

    return ParsedSource(uri=uri, scheme=scheme, authority=authority, path=path)


def source_display_name(parsed: ParsedSource) -> str:
    """A human-readable title fallback derived from a URI.

    Used when the caller supplies no explicit title. Returns the last path
    segment where there is one (the filename), otherwise the authority.
    """
    if parsed.path:
        last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if last:
            return last
    return parsed.authority
