"""The failure this suite exists for: a scanned PDF that ingests successfully.

`pypdf` reads an image-only PDF without complaint and returns an empty string
for every page. Nothing raises. The document lands in the corpus with a handful
of chunks or none, and from then on every question it should have answered comes
back with a plausible answer sourced from somewhere else. That is worse than a
rejected upload, because a rejection is a message and a hole in recall is not.

The tests below pin two things that are easy to get subtly wrong:

  * the **denominator**. ``PdfLoader`` drops pages that yield no text, so the
    number of segments is the number of pages that *worked*. Dividing by it
    reports the density of exactly the pages that were fine and waves through
    the document that is 95% images.
  * the **blast radius**. The gate is about paginated, possibly-scanned
    renderings. A Markdown file has no pages and no scanner, and a gate that
    fired on it would be rejecting documents for having short content.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import LowTextYieldError
from production_rag.ingestion import quality
from production_rag.ingestion.loaders import ExtractedSegment, load_file
from production_rag.ingestion.quality import METHOD_LOCAL
from production_rag.models.document import Document
from production_rag.models.enums import JobStatus
from tests._file_builders import make_pdf, make_scanned_pdf
from tests._jobs import drain_expecting_failure, drain_jobs

# Most of this module is pure measurement and needs no loop. The two end-to-end
# tests carry the marker individually rather than the module carrying it for
# everyone, which would warn on every synchronous test here.
module_loop = pytest.mark.asyncio(loop_scope="module")

PDF_MIME = "application/pdf"


# ─── The denominator is the whole page count, not the pages that worked ───


def test_density_divides_by_source_pages_not_extracted_segments() -> None:
    """A 20-page scan with one good page is 20 pages of document, not one."""
    pdf = make_scanned_pdf(20, text_pages=1)
    segments = load_file(pdf, "scan.pdf", PDF_MIME)

    report = quality.assess(segments, content=pdf, mime_type=PDF_MIME, method=METHOD_LOCAL)

    assert report.page_count == 20
    assert report.pages_with_text == 1
    assert report.segment_count == 1
    # The one good page is dense on its own; the document is not.
    assert report.char_count / report.segment_count > 100
    assert report.chars_per_page is not None
    assert report.chars_per_page < 50


def test_a_mostly_scanned_pdf_is_rejected() -> None:
    pdf = make_scanned_pdf(20, text_pages=1)
    segments = load_file(pdf, "scan.pdf", PDF_MIME)
    report = quality.assess(segments, content=pdf, mime_type=PDF_MIME, method=METHOD_LOCAL)

    assert not quality.passes(report, min_chars_per_page=50)


def test_a_text_pdf_is_accepted() -> None:
    pdf = make_pdf(["Real page of prose.", "Another real page."])
    segments = load_file(pdf, "report.pdf", PDF_MIME)
    report = quality.assess(segments, content=pdf, mime_type=PDF_MIME, method=METHOD_LOCAL)

    assert report.page_count == 2
    assert report.pages_with_text == 2
    assert quality.passes(report, min_chars_per_page=50)


# ─── The gate only speaks about formats that have pages ───


def test_short_markdown_is_not_judged_by_page_density() -> None:
    """No pages, no scanner, nothing for this gate to say."""
    segments = load_file(b"# Note\n\nShort.", "note.md", "text/markdown")
    report = quality.assess(
        segments, content=b"# Note\n\nShort.", mime_type="text/markdown", method=METHOD_LOCAL
    )

    assert report.page_count is None
    assert report.chars_per_page is None
    assert quality.passes(report, min_chars_per_page=50)


def test_unreadable_pdf_bytes_report_no_page_count() -> None:
    """A malformed PDF is a parse failure, not a quality failure.

    The gate must not be the thing that reports it — that would replace the
    loader's specific error with a misleading "this looks scanned".
    """
    assert quality.pdf_page_count(b"not a pdf at all") is None


# ─── The message has to be usable by whoever reads it ───


def test_rejection_names_the_numbers_that_justify_it() -> None:
    report = quality.ExtractionReport(
        method=METHOD_LOCAL,
        char_count=400,
        segment_count=2,
        page_count=40,
        pages_with_text=2,
    )

    error = quality.rejection(report, min_chars_per_page=50, remedy="Set OCR_ENABLED=true.")

    assert isinstance(error, LowTextYieldError)
    assert error.status_code == 422
    assert "10 characters per page" in error.detail
    assert "40 pages" in error.detail
    assert "minimum of 50" in error.detail
    assert "Only 2 of 40 pages" in error.detail
    assert "Set OCR_ENABLED=true." in error.detail
    # The measurements travel with the exception so a failure record can store
    # them without re-parsing the sentence above.
    assert error.report.chars_per_page == 10


# ─── End to end: the job fails and no document is created ───


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Gatekeeper", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login", json={"email": email, "password": "securepass123"}
    )
    return login.json()["access_token"]


@module_loop
async def test_scanned_upload_fails_the_job_and_writes_no_document(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    token = await _auth_token(pg_async_client, "scanned@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("scan.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )

    # Accepted at the door: the gate needs the parse, and the parse is the
    # worker's job. Cheap checks stay synchronous; this one cannot be.
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue)

    assert job.status == JobStatus.FAILED.value
    assert "characters per page" in job.error
    assert "scanned or image-only" in job.error

    # The point of the gate: nothing partial reached the corpus.
    count = await pg_session.execute(
        select(func.count())
        .select_from(Document)
        .where(Document.source == response.json()["source"])
    )
    assert count.scalar_one() == 0


@module_loop
async def test_text_pdf_still_ingests_with_the_gate_in_place(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """The gate must not be a tax on documents that were always fine."""
    token = await _auth_token(pg_async_client, "textpdf@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("report.pdf", make_pdf(["Page one.", "Page two."]), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue)

    document = await pg_session.execute(
        select(Document).where(Document.source == response.json()["source"])
    )
    stored = document.scalar_one()
    assert stored.chunk_count > 0
    # How the text was obtained is recorded, so an OCR'd corpus is auditable.
    assert stored.doc_metadata["extraction_method"] == METHOD_LOCAL


# ─── The threshold is configuration, not a constant ───


def test_threshold_is_configurable() -> None:
    pdf = make_scanned_pdf(20, text_pages=1)
    segments = load_file(pdf, "scan.pdf", PDF_MIME)
    report = quality.assess(segments, content=pdf, mime_type=PDF_MIME, method=METHOD_LOCAL)

    assert not quality.passes(report, min_chars_per_page=50)
    assert quality.passes(report, min_chars_per_page=1)
    # The shipped default, asserted against the constant rather than against
    # get_settings() — the latter reads the developer's .env, which would make
    # this pass or fail for reasons that have nothing to do with the gate.
    assert quality.DEFAULT_MIN_CHARS_PER_PAGE == 50


def test_assess_never_raises_on_an_empty_result() -> None:
    """Measuring and judging are separate so the numbers exist either way."""
    report = quality.assess(
        [ExtractedSegment(text="")], content=b"", mime_type=PDF_MIME, method=METHOD_LOCAL
    )

    assert report.char_count == 0
    assert report.page_count is None  # unreadable bytes, not "a PDF with no pages"
    assert report.as_diagnostics()["extraction_method"] == METHOD_LOCAL
