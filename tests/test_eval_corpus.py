"""Seeding the corpus and resolving gold keys back to real rows.

These run on the fast in-memory SQLite fixture: nothing here needs pgvector,
because none of it does a vector search — the point is that
``(document_key, chunk_index)`` resolves to the same text a dataset was
generated against, which is a plain relational question.
"""

from uuid import uuid4

from production_rag.eval.corpus import (
    CorpusIndex,
    corpus_files,
    corpus_key,
    corpus_source,
    read_out_of_corpus,
    seed_corpus,
)
from production_rag.eval.schema import content_sha256
from production_rag.ingestion.sources import parse_source_uri
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.ingestion_service import IngestionService

from ._jobs import hashed_embedding_service

USER_ID = uuid4()


def build_ingestion(session) -> IngestionService:
    return IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=hashed_embedding_service(),
        query_cache_repository=QueryCacheRepository(session),
        batch_size=50,
    )


def test_the_committed_seed_corpus_is_not_empty():
    files = corpus_files()
    assert files, "eval/corpus/ holds no .md files — the dataset has nothing to key against"
    assert {path.name for path in files} >= {"readme.md", "rag-production-decisions.md"}


def test_the_frozen_note_is_not_ingested_as_a_corpus_document():
    # It is a .txt precisely so the loader's *.md glob skips it.
    assert all(path.suffix == ".md" for path in corpus_files())


def test_a_corpus_source_is_a_scheme_the_worker_can_still_parse():
    # An invented scheme would ingest fine and then blow up the first time
    # anybody ran `cli reindex` over the eval user's documents.
    source = corpus_source(USER_ID, "readme.md")
    assert parse_source_uri(source).scheme == "upload"


def test_the_document_key_survives_a_change_of_owner():
    first = corpus_key(corpus_source(uuid4(), "rag-production-decisions.md"))
    second = corpus_key(corpus_source(uuid4(), "rag-production-decisions.md"))
    assert first == second == "rag-production-decisions.md"


def test_the_hand_written_out_of_corpus_questions_are_readable_and_uncommented():
    questions = read_out_of_corpus()
    assert len(questions) >= 10
    assert all(not question.startswith("#") for question in questions)
    assert all(question.endswith("?") for question in questions)


async def test_seeding_ingests_every_corpus_file_under_one_owner(test_session):
    repository = DocumentRepository(test_session)
    seeded = await seed_corpus(build_ingestion(test_session), repository, USER_ID)
    assert len(seeded) == len(corpus_files())
    assert all(item.document.user_id == USER_ID for item in seeded)
    assert all(item.document.chunk_count > 0 for item in seeded)
    assert all(item.was_new and item.reembedded for item in seeded)


async def test_seeding_twice_replaces_in_place_and_re_embeds_nothing(test_session):
    repository = DocumentRepository(test_session)
    await seed_corpus(build_ingestion(test_session), repository, USER_ID)
    first = await repository.list_documents_by_user(USER_ID, limit=100)

    # Identity is (user_id, source) and nothing about the content moved, so the
    # idempotency gate should skip the work entirely. Note the second run still
    # returns a row per file — _persist hands back the existing document on its
    # no-op path — so "did nothing" has to be read off `reembedded`, not off the
    # length of the result.
    again = await seed_corpus(build_ingestion(test_session), repository, USER_ID)
    second = await repository.list_documents_by_user(USER_ID, limit=100)

    assert len(again) == len(corpus_files())
    assert not any(item.was_new or item.reembedded for item in again)
    assert {document.id for document in first} == {document.id for document in second}


async def test_every_chunk_resolves_from_its_portable_key_with_a_matching_hash(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    repository = DocumentRepository(test_session)
    index = await CorpusIndex.load(repository, USER_ID)

    assert len(index) > 50
    for document_key, chunk_index in index.chunk_keys():
        chunk = index.get(document_key, chunk_index)
        assert chunk is not None
        # This is the tripwire the dataset relies on: the key resolves, and the
        # hash proves it resolves to the same text it was generated against.
        assert index.sha256(document_key, chunk_index) == content_sha256(chunk.content)


async def test_chunks_are_listed_in_stable_document_and_index_order(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    chunks = await DocumentRepository(test_session).list_chunks_for_owner(USER_ID, limit=10000)
    keys = [(chunk.document_id, chunk.chunk_index) for chunk in chunks]
    assert keys == sorted(keys)


async def test_chunks_are_resolved_by_document_id_and_index(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    repository = DocumentRepository(test_session)
    chunks = await repository.list_chunks_for_owner(USER_ID, limit=10)
    wanted = [(chunk.document_id, chunk.chunk_index) for chunk in chunks[:3]]

    found = await repository.get_chunks_by_keys(USER_ID, wanted)
    assert set(found) == set(wanted)
    assert found[wanted[0]].content == chunks[0].content


async def test_another_users_chunk_key_resolves_to_nothing(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    repository = DocumentRepository(test_session)
    chunks = await repository.list_chunks_for_owner(USER_ID, limit=1)
    key = (chunks[0].document_id, chunks[0].chunk_index)

    assert await repository.get_chunks_by_keys(uuid4(), [key]) == {}


async def test_an_unresolvable_key_is_absent_rather_than_raising(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    repository = DocumentRepository(test_session)
    chunks = await repository.list_chunks_for_owner(USER_ID, limit=1)
    real = (chunks[0].document_id, chunks[0].chunk_index)

    found = await repository.get_chunks_by_keys(USER_ID, [real, (chunks[0].document_id, 99999)])
    assert set(found) == {real}


async def test_asking_for_no_keys_does_not_query_at_all(test_session):
    assert await DocumentRepository(test_session).get_chunks_by_keys(USER_ID, []) == {}


async def test_the_chunk_count_matches_what_the_index_holds(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    repository = DocumentRepository(test_session)
    index = await CorpusIndex.load(repository, USER_ID)
    assert await repository.count_chunks_for_owner(USER_ID) == len(index)


async def test_the_corpus_spreads_across_documents_so_one_cannot_dominate(test_session):
    await seed_corpus(build_ingestion(test_session), DocumentRepository(test_session), USER_ID)
    index = await CorpusIndex.load(DocumentRepository(test_session), USER_ID)
    documents = {key for key, _ in index.chunk_keys()}
    assert len(documents) == len(corpus_files())
