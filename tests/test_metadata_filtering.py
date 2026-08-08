"""Metadata filtering at retrieval, over Postgres + pgvector.

Two failure classes live here, and neither one raises.

The first is a filter that does not filter. Until this change, ``filters`` was
accepted by the API, folded into the answer-cache key, and then dropped — so two
callers passing different filters got correctly-separated cache entries holding
identically unfiltered results. On a multi-tenant retrieval API that is worse
than having no parameter, because the next caller reasonably assumes that passing
``{"classification": "public"}`` does something.

The second is filtered-ANN under-return. pgvector's HNSW scan walks the graph and
applies ``WHERE`` to what it finds, so a selective filter can leave the query
returning fewer rows than ``LIMIT`` — or none — while reporting success. Recall
collapses quietly as tenants and predicates multiply. What is tested here is the
mitigation (the recall settings are really applied) and the composed behaviour (a
filtered page comes back full); reproducing the collapse itself needs a corpus
larger than a test should build, and the test that covers it says so rather than
implying otherwise.

Both need the HNSW index to exist to be exercised at all, which is why it is
declared on the model rather than only in a migration: these fixtures build their
schema with ``create_all``, so an index living only in Alembic would be absent
here and every query would fall back to an exact scan.
"""

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import ValidationError
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.ingestion.metadata import METADATA_VERSION
from production_rag.models import User
from production_rag.models.document import DocumentChunk
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import IngestionJobRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.auth_service import hash_password
from production_rag.services.document_service import DocumentService
from production_rag.services.ingestion_service import IngestionService

from ._jobs import embed_deterministically, hashed_embedding_service

pytestmark = pytest.mark.asyncio(loop_scope="module")

SPANISH_CONTRACT = (
    "CONTRATO DE PRESTACIÓN DE SERVICIOS. Fecha: 15 de marzo de 2024. "
    "De una parte la empresa ACME y de otra el contratista. Las partes acuerdan "
    "las siguientes cláusulas sobre la prestación del servicio y sus obligaciones. "
)
ENGLISH_INVOICE = (
    "INVOICE. Invoice Number: 2024-0091. Bill to Acme Corporation. "
    "Payment terms net 30 days. Subtotal 1,200.00 VAT 21% amount due 1,452.00 USD. "
)
ENGLISH_REPORT = (
    "QUARTERLY REPORT. Executive summary: revenue grew across every region. "
    "Introduction, methodology and conclusions follow in the sections below. "
)


async def _make_user(session: AsyncSession, email: str) -> User:
    user = User(
        name=email.split("@")[0],
        email=email,
        hashed_password=hash_password("pw"),
        role="user",
    )
    session.add(user)
    await session.flush()
    return user


async def _ingest(
    session: AsyncSession, user: User, name: str, body: str, repeat: int = 8
) -> None:
    """Seed through the real ingestion path, so metadata is really extracted.

    ``repeat`` sets how many chunks the document produces (~1 per 1000 chars),
    which matters only to the under-return test, where the target must own more
    matching chunks than the ``top_k`` being asked for.
    """
    await IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=hashed_embedding_service(),
        query_cache_repository=QueryCacheRepository(session),
        job_repository=IngestionJobRepository(session),
    ).ingest_now(
        title=name,
        segments=[ExtractedSegment(text=body * repeat)],
        user_id=user.id,
        source=f"upload://{user.id}/{name}.txt",
    )


def _service(session: AsyncSession) -> DocumentService:
    from unittest.mock import AsyncMock

    from production_rag.llm.client import LLMClient

    return DocumentService(
        repository=DocumentRepository(session),
        embedding_service=hashed_embedding_service(),
        llm_client=AsyncMock(spec=LLMClient),
        query_cache_repository=QueryCacheRepository(session),
    )


# ─── Metadata reaches the rows retrieval reads ───


async def test_ingestion_extracts_metadata_onto_document_and_chunks(
    pg_session: AsyncSession,
) -> None:
    """The chunk copy is what matters: the ANN query filters without a join.

    A document whose metadata lived only on ``documents`` would be unfilterable
    at the only point where filtering is worth doing.
    """
    user = await _make_user(pg_session, "extract@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)

    document = await DocumentRepository(pg_session).find_document_by_source(
        user_id=user.id, source=f"upload://{user.id}/contrato.txt"
    )
    assert document is not None
    assert document.doc_metadata["language"] == "es"
    assert document.doc_metadata["doc_type"] == "contract"
    assert document.doc_metadata["document_date"] == "2024-03-15"
    assert document.doc_metadata["extractor_version"] == METADATA_VERSION

    chunks = (
        (
            await pg_session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document.id)
            )
        )
        .scalars()
        .all()
    )
    assert chunks, "expected the document to produce chunks"
    assert all(chunk.doc_metadata == document.doc_metadata for chunk in chunks)


# ─── The filter actually filters (E4) ───


async def test_a_filtered_query_cannot_return_a_non_matching_chunk(
    pg_session: AsyncSession,
) -> None:
    """The property the dead parameter used to violate.

    Asserted over every returned row rather than by counting, because the failure
    mode being guarded against is a filter that is accepted and ignored — which
    produces a plausible-looking result set of exactly the right size.
    """
    user = await _make_user(pg_session, "filters@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)
    await _ingest(pg_session, user, "invoice", ENGLISH_INVOICE)
    await _ingest(pg_session, user, "report", ENGLISH_REPORT)

    service = _service(pg_session)

    results = await service.retrieve(
        question="cláusulas de la prestación del servicio",
        user_id=user.id,
        top_k=10,
        filters={"language": "es", "doc_type": "contract"},
        use_cache=False,
    )

    assert results, "the Spanish contract should still be reachable"
    for scored in results:
        assert scored.chunk.doc_metadata["language"] == "es"
        assert scored.chunk.doc_metadata["doc_type"] == "contract"


async def test_an_unfiltered_query_reaches_documents_the_filter_excludes(
    pg_session: AsyncSession,
) -> None:
    """The companion to the test above: proves the corpus really is mixed.

    Without this, a filter that returned only Spanish contracts because Spanish
    contracts were all that existed would look identical to a working one.
    """
    user = await _make_user(pg_session, "unfiltered@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)
    await _ingest(pg_session, user, "invoice", ENGLISH_INVOICE)

    results = await _service(pg_session).retrieve(
        question="payment terms and amounts due",
        user_id=user.id,
        top_k=10,
        use_cache=False,
    )

    languages = {scored.chunk.doc_metadata.get("language") for scored in results}
    assert languages == {"es", "en"}


async def test_a_filter_matching_nothing_returns_nothing_not_everything(
    pg_session: AsyncSession,
) -> None:
    """An unmatched filter must narrow to empty, never fall open.

    A filter that silently degraded to "no filter" on an unknown key is how a
    governance predicate turns into a data leak.
    """
    user = await _make_user(pg_session, "nomatch@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)

    service = _service(pg_session)

    unknown_value = await service.retrieve(
        question="cláusulas", user_id=user.id, top_k=10,
        filters={"language": "ja"}, use_cache=False,
    )
    unknown_key = await service.retrieve(
        question="cláusulas", user_id=user.id, top_k=10,
        filters={"classification": "top-secret"}, use_cache=False,
    )

    assert unknown_value == []
    assert unknown_key == []


async def test_absent_keys_do_not_match_a_filter_on_that_key(
    pg_session: AsyncSession,
) -> None:
    """Undetermined is not "matches".

    Extraction omits keys it could not determine rather than storing null, and
    rows predating extraction hold ``{}``. Containment must exclude both from a
    filter on that key — the alternative is a document leaking into a filtered
    result precisely because detection failed on it.
    """
    user = await _make_user(pg_session, "absent@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)

    # Strip the extracted metadata back to what a pre-extraction row looks like.
    await pg_session.execute(
        text("UPDATE document_chunks SET metadata = '{}'::jsonb WHERE owner_id = :uid"),
        {"uid": user.id},
    )

    results = await _service(pg_session).retrieve(
        question="cláusulas", user_id=user.id, top_k=10,
        filters={"language": "es"}, use_cache=False,
    )
    assert results == []


@pytest.mark.parametrize("filters", [None, {}])
async def test_an_empty_filter_is_equivalent_to_no_filter(
    pg_session: AsyncSession, filters: dict | None
) -> None:
    """``metadata @> '{}'`` matches every row, so the two must agree."""
    user = await _make_user(pg_session, f"empty{filters!s:.4}@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)
    await _ingest(pg_session, user, "invoice", ENGLISH_INVOICE)

    service = _service(pg_session)
    baseline = await service.retrieve(
        question="terms", user_id=user.id, top_k=10, use_cache=False
    )
    filtered = await service.retrieve(
        question="terms", user_id=user.id, top_k=10, filters=filters, use_cache=False
    )

    assert [s.chunk.id for s in filtered] == [s.chunk.id for s in baseline]


async def test_filtering_does_not_widen_the_tenant_boundary(
    pg_session: AsyncSession,
) -> None:
    """Owner scope and the metadata filter are ANDed, never traded off.

    Both predicates now share one query, so a bug that reordered or replaced them
    would be a cross-tenant read — the most expensive failure in this file.
    """
    owner = await _make_user(pg_session, "owner@example.com")
    other = await _make_user(pg_session, "other@example.com")
    await _ingest(pg_session, owner, "contrato", SPANISH_CONTRACT)
    await _ingest(pg_session, other, "contrato", SPANISH_CONTRACT)

    results = await _service(pg_session).retrieve(
        question="cláusulas de la prestación",
        user_id=owner.id,
        top_k=10,
        filters={"doc_type": "contract"},
        use_cache=False,
    )

    assert results, "the owner's own contract should match"
    assert all(scored.chunk.owner_id == owner.id for scored in results)


async def test_a_malformed_filter_is_rejected_before_it_reaches_the_query(
    pg_session: AsyncSession,
) -> None:
    """retrieve() validates too, not just the request schema.

    The agent tools and eval harnesses call it directly, without passing through
    ``QueryRequest``.
    """
    user = await _make_user(pg_session, "malformed@example.com")

    with pytest.raises(ValidationError):
        await _service(pg_session).retrieve(
            question="x", user_id=user.id, filters={"meta": {"nested": True}}, use_cache=False
        )


# ─── Filtered-ANN under-return (D3) ───


async def test_the_hnsw_recall_settings_are_actually_in_force(
    pg_session: AsyncSession,
) -> None:
    """The mitigation is two ``SET LOCAL``s, and they are allowed to fail softly.

    That combination is its own hazard: an unsupported setting is meant to log and
    continue, so a *malformed* one degrades identically — silently, with recall
    quietly back to the unmitigated behaviour and nothing in the result set
    looking wrong. (This is not hypothetical. ``SET LOCAL hnsw.ef_search = :ef``
    is a syntax error, because SET takes a literal and never a bind parameter, and
    it failed exactly this invisibly.)

    So the assertion is on the server's own view of the settings, read back after
    a real search, rather than on anything the query returned.
    """
    user = await _make_user(pg_session, "gucs@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)

    await _service(pg_session).retrieve(
        question="cláusulas", user_id=user.id, top_k=3, use_cache=False
    )

    assert await pg_session.scalar(text("SELECT current_setting('hnsw.ef_search')")) == "100"
    assert (
        await pg_session.scalar(text("SELECT current_setting('hnsw.iterative_scan')"))
        == "relaxed_order"
    )


async def test_a_filtered_search_returns_a_full_page_across_many_tenants(
    pg_session: AsyncSession,
) -> None:
    """A selective filter must not eat the result page.

    pgvector applies ``WHERE`` to what the HNSW walk finds rather than before it,
    so a query whose filter excludes most of the index can return fewer rows than
    ``LIMIT`` while reporting success. This asserts the composed behaviour — owner
    scope plus two metadata predicates, over an index dominated by rows that match
    neither — still yields a full page.

    **Honest limit: at this corpus size the query returns a full page with the
    iterative scan disabled too**, so this is a regression guard, not a
    reproduction. The collapse needs an index far larger than a test should build
    (the walk has to miss ``ef_search`` candidates in a row, which takes tens of
    thousands of vectors). What *is* verified directly is that the mitigating
    settings are applied — see the test above — and D3's real proof stays what the
    decisions doc specifies: recall measured against a synthetic multi-tenant
    corpus, outside this suite.

    ``enable_seqscan = off`` forces the index path; otherwise the planner picks an
    exact sequential scan at this size and the ANN behaviour is never exercised.
    """
    top_k = 5
    tenants = [await _make_user(pg_session, f"ann{i}@example.com") for i in range(15)]
    target = tenants[0]

    # The target's document is Spanish, so a metadata predicate selects it, and
    # long enough to hold more than top_k chunks — otherwise "returned fewer than
    # top_k" would just mean "there were fewer than top_k".
    await _ingest(pg_session, target, "contrato", SPANISH_CONTRACT, repeat=40)
    # Everyone else's is English filler, crowding the graph around the target.
    for i, tenant in enumerate(tenants[1:], start=1):
        await _ingest(pg_session, tenant, f"report{i}", ENGLISH_REPORT + f"Region {i}. ", repeat=40)

    total = await pg_session.scalar(select(func.count()).select_from(DocumentChunk))
    matching = await pg_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(
            DocumentChunk.owner_id == target.id,
            text("metadata @> '{\"language\":\"es\",\"doc_type\":\"contract\"}'::jsonb"),
        )
    )
    assert total is not None and matching is not None
    assert matching > top_k, "the filter must match more chunks than are requested"
    assert total > matching * 5, "the index must be dominated by rows the filter excludes"

    await pg_session.execute(text("SET LOCAL enable_seqscan = off"))

    results = await _service(pg_session).retrieve(
        question=SPANISH_CONTRACT[:80],
        user_id=target.id,
        top_k=top_k,
        filters={"language": "es", "doc_type": "contract"},
        use_cache=False,
    )

    # The assertion is about *count*: a short result set is the failure, and it
    # looks exactly like a correct one otherwise.
    assert len(results) == top_k, (
        f"filtered ANN under-returned: got {len(results)} of {top_k} requested "
        f"({matching} chunks match the filter out of {total} indexed)"
    )
    assert all(scored.chunk.owner_id == target.id for scored in results)


async def test_the_nearest_chunk_to_its_own_text_ranks_first(
    pg_session: AsyncSession,
) -> None:
    """Guards the filtered query's ordering.

    ``relaxed_order`` lets pgvector return rows in approximate distance order, so
    the repository re-sorts outside the scan. Without that, ``ScoredChunk.score``
    — what an eval harness ranks on — would be subtly out of order.
    """
    user = await _make_user(pg_session, "ranking@example.com")
    await _ingest(pg_session, user, "contrato", SPANISH_CONTRACT)
    await _ingest(pg_session, user, "invoice", ENGLISH_INVOICE)
    await _ingest(pg_session, user, "report", ENGLISH_REPORT)

    chunk = (
        await pg_session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.owner_id == user.id, DocumentChunk.document_title == "invoice")
            .limit(1)
        )
    ).scalar_one()

    service = _service(pg_session)
    # Query with the chunk's own text: its vector is identical, so distance ~0.
    service._embedding_service.embed_text.side_effect = lambda _: embed_deterministically(
        chunk.content
    )

    results = await service.retrieve(
        question="ignored — the embedder is pinned to the chunk's own vector",
        user_id=user.id,
        top_k=5,
        use_cache=False,
    )

    assert results[0].chunk.id == chunk.id
    scores = [scored.score for scored in results]
    assert scores == sorted(scores, reverse=True)
