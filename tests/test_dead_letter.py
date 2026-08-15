"""What happens to a document that never makes it in.

Before this, a failed ingestion left one free-text sentence on the job row,
overwritten by the next attempt. Two things were unavailable as a result: the
*history* of an attempt sequence, and any answer to "what is failing, and how
often" — you cannot GROUP BY a message with a filename in it.

The tests here pin the two behaviours that make the dead-letter table more than
a log:

  * **a non-retryable failure is terminal on the first attempt.** A scanned PDF
    is still a scanned PDF on attempt three. Retrying it burns a worker slot and,
    once OCR is in the path, real money — so the queue is told not to.
  * **a retryable failure is not.** An origin that 502s might not 502 next time,
    and giving up on it would lose a document to a transient problem.

Getting that boundary backwards is silent in both directions: retry-forever
looks like a slow queue, give-up-immediately looks like a flaky corpus.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    InvalidSourceURIError,
    LowTextYieldError,
    SourceFetchError,
    UnsupportedFileTypeError,
    ValidationError,
)
from production_rag.ingestion.failures import (
    FailureReason,
    FailureStage,
    classify,
    diagnostics,
)
from production_rag.ingestion.quality import METHOD_LOCAL, ExtractionReport
from production_rag.models.enums import JobStatus
from production_rag.models.failed_ingestion import FailedIngestion
from tests._file_builders import make_scanned_pdf
from tests._jobs import drain_expecting_failure

module_loop = pytest.mark.asyncio(loop_scope="module")

PDF_MIME = "application/pdf"


# ─── Classification: the retryable boundary ───


@pytest.mark.parametrize(
    ("exc", "reason", "stage"),
    [
        (
            LowTextYieldError("scanned", report=ExtractionReport(METHOD_LOCAL, 0, 0)),
            FailureReason.LOW_TEXT_YIELD,
            FailureStage.QUALITY_GATE,
        ),
        (
            UnsupportedFileTypeError("nope"),
            FailureReason.UNSUPPORTED_TYPE,
            FailureStage.PARSE,
        ),
        (SourceFetchError("502"), FailureReason.FETCH_FAILED, FailureStage.FETCH),
        (
            ConnectorNotConfiguredError("no key"),
            FailureReason.OCR_NOT_CONFIGURED,
            FailureStage.OCR,
        ),
        (InvalidSourceURIError("bad"), FailureReason.INVALID_SOURCE, FailureStage.FETCH),
        (ValidationError("generic"), FailureReason.PARSE_ERROR, FailureStage.PARSE),
        (RuntimeError("boom"), FailureReason.INTERNAL, FailureStage.UNKNOWN),
    ],
)
def test_classification_maps_the_taxonomy(exc, reason, stage) -> None:
    classification = classify(exc)
    assert classification.reason is reason
    assert classification.stage is stage


def test_only_upstream_failures_are_retryable() -> None:
    """Retrying is for failures whose answer could plausibly differ next time."""
    assert classify(SourceFetchError("origin 502")).retryable
    # An unrecognised exception is retried: most likely a dropped connection,
    # and wrongly retrying costs one attempt while wrongly giving up loses a
    # document.
    assert classify(RuntimeError("who knows")).retryable

    assert not classify(UnsupportedFileTypeError("x")).retryable
    assert not classify(
        LowTextYieldError("scanned", report=ExtractionReport(METHOD_LOCAL, 0, 0))
    ).retryable
    assert not classify(ConnectorNotConfiguredError("x")).retryable


def test_subclass_ordering_is_specific_first() -> None:
    """Both of these are ValidationError; neither may collapse into it."""
    assert classify(LowTextYieldError("x", report=ExtractionReport(METHOD_LOCAL, 0, 0))).reason is (
        FailureReason.LOW_TEXT_YIELD
    )
    assert classify(InvalidSourceURIError("x")).reason is FailureReason.INVALID_SOURCE


def test_gate_measurements_travel_with_the_exception() -> None:
    """chars-per-page is stored, so counting low-density documents is a query."""
    report = ExtractionReport(
        method=METHOD_LOCAL, char_count=400, segment_count=2, page_count=40, pages_with_text=2
    )
    captured = diagnostics(LowTextYieldError("scanned", report=report))

    assert captured["chars_per_page"] == 10
    assert captured["page_count"] == 40
    assert captured["pages_with_text"] == 2
    # An exception carrying no report contributes no keys rather than nulls.
    assert diagnostics(RuntimeError("boom")) == {}


# ─── End to end ───


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Operator", "email": email, "password": "securepass123"},
    )
    login = await client.post("/auth/login", json={"email": email, "password": "securepass123"})
    return login.json()["access_token"]


async def _failures_for(session: AsyncSession, job_id) -> list[FailedIngestion]:
    result = await session.execute(
        select(FailedIngestion)
        .where(FailedIngestion.job_id == job_id)
        .order_by(FailedIngestion.created_at)
    )
    return list(result.scalars().all())


@module_loop
async def test_a_scanned_pdf_is_terminal_on_the_first_attempt(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    token = await _auth_token(pg_async_client, "dl-scan@example.com")
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("scan.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue)

    assert job.status == JobStatus.FAILED.value
    assert job.attempts == 1
    # The countable form of the same failure, next to the readable one.
    assert job.failure_reason == FailureReason.LOW_TEXT_YIELD.value
    assert "characters per page" in job.error

    records = await _failures_for(pg_session, job.id)
    assert len(records) == 1
    failure = records[0]
    assert failure.reason == FailureReason.LOW_TEXT_YIELD.value
    assert failure.stage == FailureStage.QUALITY_GATE.value
    assert failure.attempt == 1
    # Terminal after ONE attempt, not three: nothing about attempt two would
    # make this PDF less scanned.
    assert failure.is_terminal is True
    assert failure.diagnostics["page_count"] == 20
    assert failure.diagnostics["chars_per_page"] < 50
    # Denormalized, so the row still says what it was about if the job goes.
    assert failure.source == response.json()["source"]
    assert failure.filename == "scan.pdf"


@module_loop
async def test_a_retryable_failure_is_not_terminal_until_attempts_run_out(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """A URL that cannot be fetched is upstream's problem, and may not stay one."""
    token = await _auth_token(pg_async_client, "dl-fetch@example.com")
    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "https://127.0.0.1:9/never-served.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue)

    assert job.failure_reason == FailureReason.FETCH_FAILED.value
    failure = (await _failures_for(pg_session, job.id))[0]
    assert failure.stage == FailureStage.FETCH.value
    # Attempt 1 of 3 — the queue should try again.
    assert failure.is_terminal is False


@module_loop
async def test_each_attempt_appends_rather_than_overwriting(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """The history the single `error` column used to destroy."""
    token = await _auth_token(pg_async_client, "dl-history@example.com")
    await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "https://127.0.0.1:9/still-never-served.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    job_id = job_queue.enqueued[0]
    job_queue.enqueued.append(job_id)  # a second delivery, as a retry would be
    first = await drain_expecting_failure(pg_session, job_queue)
    second = await drain_expecting_failure(pg_session, job_queue)
    assert first.id == second.id

    records = await _failures_for(pg_session, job_id)
    assert [record.attempt for record in records] == [1, 2]
    assert [record.is_terminal for record in records] == [False, False]


# ─── The operator surface ───


@module_loop
async def test_failures_endpoint_lists_only_the_callers_terminal_failures(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mine = await _auth_token(pg_async_client, "dl-mine@example.com")
    theirs = await _auth_token(pg_async_client, "dl-theirs@example.com")

    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("mine.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {mine}"},
    )
    await drain_expecting_failure(pg_session, job_queue)

    listed = await pg_async_client.get(
        "/documents/failures", headers={"Authorization": f"Bearer {mine}"}
    )
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 1
    assert body[0]["reason"] == FailureReason.LOW_TEXT_YIELD.value
    assert body[0]["is_terminal"] is True
    assert body[0]["diagnostics"]["page_count"] == 20

    # Someone else's failure is absent, not forbidden — the endpoint does not
    # confirm that an id exists.
    other = await pg_async_client.get(
        "/documents/failures", headers={"Authorization": f"Bearer {theirs}"}
    )
    assert other.json() == []


@module_loop
async def test_terminal_only_filter_hides_in_flight_failures(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    token = await _auth_token(pg_async_client, "dl-filter@example.com")
    await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "https://127.0.0.1:9/transient.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    await drain_expecting_failure(pg_session, job_queue)

    headers = {"Authorization": f"Bearer {token}"}
    # Default asks "what needs me?" — a job awaiting retry does not.
    assert await _json(pg_async_client, "/documents/failures", headers) == []
    history = await _json(pg_async_client, "/documents/failures?terminal_only=false", headers)
    assert len(history) == 1
    assert history[0]["reason"] == FailureReason.FETCH_FAILED.value


async def _json(client: AsyncClient, url: str, headers: dict[str, str]):
    response = await client.get(url, headers=headers)
    assert response.status_code == 200
    return response.json()


@module_loop
async def test_retry_requeues_the_job_and_keeps_its_progress(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """Cheap because a failed job keeps its payload — that is why it is kept."""
    token = await _auth_token(pg_async_client, "dl-retry@example.com")
    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("retry.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    job = await drain_expecting_failure(pg_session, job_queue)
    job.processed_chunks = 5  # stand in for work committed before the failure
    await pg_session.commit()

    failures = await _json(
        pg_async_client, "/documents/failures", {"Authorization": f"Bearer {token}"}
    )
    response = await pg_async_client.post(
        f"/documents/failures/{failures[0]['failure_id']}/retry",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await pg_session.refresh(job)
    assert job.status == JobStatus.PENDING.value
    assert job.error is None
    assert job.failure_reason is None
    # A human retrying after a fix deserves the full attempt budget...
    assert job.attempts == 0
    # ...but not a re-embed of work that is already in the database.
    assert job.processed_chunks == 5
    assert job.id in job_queue.enqueued


@module_loop
async def test_retry_conflicts_when_there_is_nothing_to_retry_from(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """An upload:// job with no staged bytes cannot be re-driven, only re-uploaded."""
    token = await _auth_token(pg_async_client, "dl-nopayload@example.com")
    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("gone.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    job = await drain_expecting_failure(pg_session, job_queue)
    job.payload = None  # as mark_succeeded would have left it
    await pg_session.commit()

    failures = await _json(
        pg_async_client, "/documents/failures", {"Authorization": f"Bearer {token}"}
    )
    response = await pg_async_client.post(
        f"/documents/failures/{failures[0]['failure_id']}/retry",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 409
    assert "Re-upload the document" in response.json()["detail"]
