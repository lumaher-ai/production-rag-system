"""Source-URI construction and parsing.

``source`` is half the ``(user_id, source)`` ingestion identity, so these are
identity tests, not string-formatting tests: a URI that round-trips wrong makes
a re-upload duplicate a document instead of replacing it.
"""

from uuid import UUID, uuid4

import pytest

from production_rag.exceptions import InvalidSourceURIError
from production_rag.ingestion.sources import (
    SOURCE_URI_MAX_LEN,
    build_upload_uri,
    parse_source_uri,
    source_display_name,
)

USER_ID = UUID("11111111-2222-3333-4444-555555555555")


def test_build_upload_uri_has_scheme_owner_and_filename() -> None:
    uri = build_upload_uri(USER_ID, "reporte.pdf")
    assert uri == f"upload://{USER_ID}/reporte.pdf"


def test_build_upload_uri_percent_encodes_path_separators() -> None:
    """A filename must not be able to forge extra URI structure."""
    uri = build_upload_uri(USER_ID, "a/b?c#d.pdf")

    parsed = parse_source_uri(uri)
    assert parsed.authority == str(USER_ID)
    # Decoded back to exactly one path segment, not three.
    assert parsed.path == "a/b?c#d.pdf"
    assert "%2F" in uri


def test_build_upload_uri_is_stable_across_calls() -> None:
    """Identity depends on this: the same file must mint the same URI."""
    assert build_upload_uri(USER_ID, "réport ñ.pdf") == build_upload_uri(USER_ID, "réport ñ.pdf")


def test_build_upload_uri_differs_per_owner() -> None:
    assert build_upload_uri(uuid4(), "x.pdf") != build_upload_uri(uuid4(), "x.pdf")


def test_build_upload_uri_handles_missing_filename() -> None:
    assert build_upload_uri(USER_ID, None) == f"upload://{USER_ID}/document"
    assert build_upload_uri(USER_ID, "   ") == f"upload://{USER_ID}/document"


def test_build_upload_uri_trims_to_column_width_and_stays_decodable() -> None:
    """Over-long names trim the filename, never the scheme or owner."""
    uri = build_upload_uri(USER_ID, "ñ" * 2000)

    assert len(uri) <= SOURCE_URI_MAX_LEN
    assert uri.startswith(f"upload://{USER_ID}/")
    # The critical property: trimming must not sever a percent-triplet.
    parse_source_uri(uri)


@pytest.mark.parametrize(
    ("uri", "scheme", "authority", "path"),
    [
        (f"upload://{USER_ID}/notes.txt", "upload", str(USER_ID), "notes.txt"),
        ("https://example.com/docs/report.pdf", "https", "example.com", "docs/report.pdf"),
        ("gdrive://1A2b3C4d5E", "gdrive", "1A2b3C4d5E", ""),
    ],
)
def test_parse_source_uri_splits_known_schemes(
    uri: str, scheme: str, authority: str, path: str
) -> None:
    parsed = parse_source_uri(uri)
    assert (parsed.scheme, parsed.authority, parsed.path) == (scheme, authority, path)


def test_parse_source_uri_marks_only_connector_schemes_fetchable() -> None:
    # upload:// names bytes that arrived over multipart — nothing to fetch.
    assert parse_source_uri(f"upload://{USER_ID}/a.txt").is_fetchable is False
    assert parse_source_uri("https://example.com/a.pdf").is_fetchable is True
    assert parse_source_uri("gdrive://abc").is_fetchable is True


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "reporte.pdf",  # the legacy bare filename this change replaces
        "/var/data/report.pdf",
        "ftp://example.com/report.pdf",
        "s3://bucket/key.pdf",  # a plausible scheme we deliberately do not ship
        "https://",
        "https://example.com",  # host with no document path
    ],
)
def test_parse_source_uri_rejects_malformed_or_unsupported(bad: str) -> None:
    with pytest.raises(InvalidSourceURIError):
        parse_source_uri(bad)


def test_parse_source_uri_rejects_over_length() -> None:
    with pytest.raises(InvalidSourceURIError):
        parse_source_uri("https://example.com/" + "a" * SOURCE_URI_MAX_LEN)


def test_source_display_name_prefers_filename() -> None:
    assert source_display_name(parse_source_uri("https://example.com/a/report.pdf")) == "report.pdf"
    assert source_display_name(parse_source_uri("gdrive://1A2b3C")) == "1A2b3C"
