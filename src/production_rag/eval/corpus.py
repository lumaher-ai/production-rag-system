"""Getting the seed corpus into the database, and reading chunks back out.

The corpus is ingested through the real pipeline — ``IngestionService.ingest_now``,
the same ``_persist`` path a worker runs — rather than by inserting rows. A
dataset generated from chunks that were produced differently from production
chunks measures a system nobody runs.
"""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from uuid import UUID

from production_rag.config import PROJECT_ROOT
from production_rag.eval.sampling import ChunkRef
from production_rag.eval.schema import content_sha256
from production_rag.ingestion.loaders import load_file
from production_rag.ingestion.sources import build_upload_uri
from production_rag.logging_config import get_logger
from production_rag.models.document import Document, DocumentChunk
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.services.ingestion_service import IngestionService

logger = get_logger(__name__)

EVAL_DIR = PROJECT_ROOT / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
SILVER_PATH = EVAL_DIR / "dataset.silver.jsonl"
REJECTED_PATH = EVAL_DIR / "dataset.rejected.jsonl"
CURATED_PATH = EVAL_DIR / "dataset.jsonl"
OUT_OF_CORPUS_PATH = EVAL_DIR / "out_of_corpus.txt"
AUDIT_DIR = EVAL_DIR / "audit"
RUNS_DIR = EVAL_DIR / "runs"

def corpus_source(user_id: UUID, filename: str) -> str:
    """The document ``source`` a seeded corpus file is stored under.

    A plain ``upload://`` URI, minted by the same helper the multipart route
    uses. Inventing a scheme of its own was the obvious move and is wrong: the
    worker validates every source it re-materializes against
    ``sources.KNOWN_SCHEMES``, so an ``evalcorpus://`` document would ingest
    cleanly and then blow up the day somebody ran ``cli reindex`` — a landmine
    laid in production code paths for the convenience of a test fixture.
    ``upload://`` is already the "bytes that exist nowhere else" scheme, which
    is exactly what a frozen snapshot is, and it takes the stored-body re-index
    path for free.
    """
    return build_upload_uri(user_id, filename)


def corpus_key(source: str) -> str:
    """The dataset's portable document key: the corpus filename.

    Not the ``source`` URI, because ``upload://`` embeds the owner's UUID and
    that differs on every machine that seeds the corpus — and the dataset is
    committed, so it must resolve against a database somebody else built. Not
    ``document_id`` either, for the same reason: it is a ``uuid4`` minted at
    creation. The filename is the one identifier that survives a fresh checkout,
    a fresh database and a fresh user.
    """
    return unquote(source.rsplit("/", 1)[-1])


def corpus_files(directory: Path = CORPUS_DIR) -> list[Path]:
    """The frozen snapshots, in a stable order. ``.txt`` notes are not corpus."""
    return sorted(directory.glob("*.md"))


@dataclass(frozen=True, slots=True)
class SeededDocument:
    """One corpus file after seeding, and whether it cost anything.

    ``reembedded`` is derived here rather than read off the return value because
    ``IngestionService._persist`` returns the existing row on its no-op path
    just as it does after real work — the two are indistinguishable to a caller
    that only looks at what came back. Comparing the content hash captured
    *before* the call is what separates them, and the distinction matters: a
    seeding run that silently re-embedded the whole corpus is a bill, and one
    that silently did nothing when you expected work is a bug.
    """

    document: Document
    was_new: bool
    reembedded: bool


async def seed_corpus(
    ingestion: IngestionService,
    documents: DocumentRepository,
    user_id: UUID,
    directory: Path = CORPUS_DIR,
) -> list[SeededDocument]:
    """Ingest every frozen snapshot under one owner, replacing in place.

    Idempotent by way of the pipeline's own idempotency: ``(user_id, source)``
    is a document's identity, and an unchanged body whose normalizer, chunker
    and embedding model all still match is skipped without re-embedding. So
    re-running this after a code change costs nothing unless something that
    actually affects the vectors moved — which is exactly when it should.
    """
    wanted = {path.name for path in corpus_files(directory)}
    foreign = [
        document
        for document in await documents.list_documents_by_user(user_id, limit=1000)
        if corpus_key(document.source) not in wanted
    ]
    if foreign:
        # Not fatal, but it makes every retrieval-based step non-reproducible:
        # search is scoped by owner_id and nothing finer, so the unanswerable
        # verifier and the baseline will both range over these documents too.
        logger.warning(
            "eval_corpus_owner_has_foreign_documents",
            count=len(foreign),
            titles=[document.title for document in foreign][:5],
            consequence="unanswerable verification and baseline scores will not be reproducible",
        )

    seeded: list[SeededDocument] = []
    for path in corpus_files(directory):
        source = corpus_source(user_id, path.name)
        prior = await documents.find_document_by_source(user_id=user_id, source=source)
        # Captured as a value, not held as a reference: a replace mutates the
        # same ORM row in place, so reading it afterwards would compare the new
        # hash against itself and report "unchanged" for every run.
        prior_hash = prior.content_hash if prior is not None else None

        segments = load_file(path.read_bytes(), path.name, None)
        document = await ingestion.ingest_now(
            title=path.stem,
            segments=segments,
            user_id=user_id,
            source=source,
        )
        if document is None:
            logger.warning("eval_corpus_not_ingested", source=source)
            continue

        record = SeededDocument(
            document=document,
            was_new=prior is None,
            reembedded=prior is None or prior_hash != document.content_hash,
        )
        logger.info(
            "eval_corpus_seeded",
            source=source,
            chunks=document.chunk_count,
            was_new=record.was_new,
            reembedded=record.reembedded,
        )
        seeded.append(record)
    return seeded


class CorpusIndex:
    """Every eval chunk, in memory, addressed by ``(document_key, chunk_index)``.

    The corpus is ~200 chunks of a few hundred tokens each, so holding all of it
    costs a few megabytes and saves the generator, the gates, the audit renderer
    and the baseline from each re-querying for the same text. It also gives the
    overlap gate cheap access to a chunk's neighbours, which is otherwise an N+1
    the size of the dataset.
    """

    def __init__(self, chunks: list[DocumentChunk], keys_by_document: dict[UUID, str]) -> None:
        self._by_key: dict[tuple[str, int], DocumentChunk] = {}
        self._key_by_document = keys_by_document
        self._document_by_key = {key: doc_id for doc_id, key in keys_by_document.items()}
        for chunk in chunks:
            key = keys_by_document.get(chunk.document_id)
            if key is None:
                continue
            self._by_key[(key, chunk.chunk_index)] = chunk

    @classmethod
    async def load(
        cls,
        repository: DocumentRepository,
        user_id: UUID,
        directory: Path = CORPUS_DIR,
    ) -> "CorpusIndex":
        """Load only the seed corpus, even when the owner has other documents.

        Restricting to the frozen filenames is not tidiness — it is what keeps
        the dataset reproducible. Retrieval is scoped by ``owner_id`` and nothing
        finer, so an eval user who also holds unrelated uploads would otherwise
        have questions generated against documents nobody else can obtain, gold
        keys that resolve on one machine and nowhere else, and a baseline whose
        denominator counts chunks the dataset never covered.

        The remaining exposure is real and cannot be fixed here: the *retrieval*
        used to verify unanswerables and to score the baseline still searches
        everything the owner has. Give the corpus its own user (``EVAL_USER_EMAIL``)
        and that goes away too; ``seed-corpus`` warns when it does not.
        """
        wanted = {path.name for path in corpus_files(directory)}
        documents = await repository.list_documents_by_user(user_id, limit=1000)
        keys = {
            document.id: corpus_key(document.source)
            for document in documents
            if corpus_key(document.source) in wanted
        }
        chunks = await repository.list_chunks_for_owner(user_id, limit=20000)
        return cls(chunks, keys)

    def __len__(self) -> int:
        return len(self._by_key)

    def chunk_keys(self) -> list[tuple[str, int]]:
        return sorted(self._by_key)

    def get(self, document_key: str, chunk_index: int) -> DocumentChunk | None:
        return self._by_key.get((document_key, chunk_index))

    def key_for(self, document_id: UUID) -> str | None:
        return self._key_by_document.get(document_id)

    def document_id_for(self, document_key: str) -> UUID | None:
        return self._document_by_key.get(document_key)

    def content(self, document_key: str, chunk_index: int) -> str | None:
        chunk = self.get(document_key, chunk_index)
        return chunk.content if chunk else None

    def sha256(self, document_key: str, chunk_index: int) -> str | None:
        chunk = self.get(document_key, chunk_index)
        return content_sha256(chunk.content) if chunk else None

    def neighbours(self, document_key: str, chunk_index: int) -> list[tuple[int, str]]:
        """The chunks either side, for the overlap gate."""
        found: list[tuple[int, str]] = []
        for offset in (-1, 1):
            chunk = self.get(document_key, chunk_index + offset)
            if chunk is not None:
                found.append((chunk.chunk_index, chunk.content))
        return found

    def refs(self) -> list[ChunkRef]:
        """The sampler's view: identity and size, no text, no vectors."""
        return [
            ChunkRef(
                document_key=key,
                chunk_index=index,
                token_count=chunk.token_count,
                document_title=chunk.document_title,
            )
            for (key, index), chunk in sorted(self._by_key.items())
        ]


def read_out_of_corpus(path: Path = OUT_OF_CORPUS_PATH) -> list[str]:
    """The hand-written off-domain questions. Comments and blanks ignored."""
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
