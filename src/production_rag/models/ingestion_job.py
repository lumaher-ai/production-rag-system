from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from production_rag.database import Base
from production_rag.models.enums import JobStatus


class IngestionJob(Base):
    """A unit of ingestion work, durable across process restarts.

    Ingestion used to run inside the HTTP request: parse, embed every chunk, and
    write, all in one transaction. A large document held the connection and the
    transaction open for a minute, and a failure part-way discarded every
    embedding already paid for. This row is what makes the work outlive the
    request — the endpoint records the job and returns, and a worker drives it.

    The row is both the **queue's payload** (what to ingest) and the **progress
    record** (how far it got), so a caller polling for status and a worker
    resuming after a crash read the same state.
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Owner scope. Indexed because the status endpoints always filter by it —
    # a job is only ever readable by the user who created it.
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=JobStatus.PENDING.value,
        nullable=False,
        index=True,
    )

    # ─── What to ingest ───
    # The canonical source URI (see ingestion/sources.py). For upload:// the
    # bytes live in `payload`; for https:// and gdrive:// the worker re-fetches
    # from this URI, so no staging is needed.
    source: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Staged upload bytes. A worker is a different process and cannot read the
    # request's UploadFile, so the bytes have to be handed over somewhere both
    # can reach. Postgres keeps it transactional with the job row and adds no
    # infrastructure; it is cleared on success so large blobs do not accumulate.
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Loader dispatch hints, captured at request time because the worker has no
    # other way to recover them for staged bytes.
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ─── Progress ───
    # Known only after parsing and chunking, hence nullable.
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Doubles as the **resume cursor**: chunking is deterministic for a given
    # (content, chunker_version), so a retry re-derives the same chunk list and
    # skips this many. That is why resuming needs no persisted chunk text.
    processed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # The chunker the completed work was produced under. If it changes between
    # attempts the cursor is meaningless — resuming would splice two chunkings
    # together — so the worker restarts the job from zero instead.
    chunker_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ─── Outcome ───
    # SET NULL rather than CASCADE: deleting a document should not erase the
    # record that it was once ingested.
    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Liveness signal, refreshed when the job is claimed and after every batch.
    # A worker killed outright (SIGKILL, OOM, pod eviction) never runs its
    # failure handler, so its job would otherwise sit in 'running' forever with
    # nothing to retry it. The sweeper uses a stale heartbeat to tell "died" from
    # "still working", which a single started_at timestamp cannot express.
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
