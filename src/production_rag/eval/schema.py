"""The dataset record — the shape everything else agrees on.

Two decisions are encoded here and both are load-bearing.

**Gold context is keyed on ``(document_key, chunk_index)``, never on
``DocumentChunk.id``.** Chunk ids are ``uuid4`` and every re-ingest deletes and
re-creates the rows, so a dataset keyed on them points at nothing the morning
after a ``reindex``. Nor is ``document_id`` the portable key: it too is a
``uuid4`` minted at creation, so it survives a re-ingest (identity is
``(user_id, source)``, replace-in-place) but *not* a fresh database — and this
dataset is committed, so somebody re-seeding a clean checkout must be able to
resolve it. Nor is the ``source`` URI: ``upload://`` embeds the owner's UUID in
its authority segment, which differs on every machine. ``document_key`` is the
corpus filename — the one identifier that survives a fresh checkout, a fresh
database and a fresh user. ``document_id`` is carried alongside for the local
session and is explicitly not authoritative.

**Every gold entry carries the sha256 of its chunk's content.** The key resolves
whether or not the chunker moved; only the hash reveals that it now resolves to
*different text*. Without it a ``CHUNKER_VERSION`` bump silently re-points every
question in the file at whatever landed at that index instead, and the metrics
keep reporting numbers.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Bump on ANY breaking change to the record shape below. A reader that meets an
# unknown version raises rather than silently mis-reading fields it recognises —
# a half-understood eval record produces a plausible number, which is worse than
# a crash.
SCHEMA_VERSION = 1

QueryType = Literal["paraphrase", "exact_term", "multi_hop", "unanswerable"]
UnanswerableKind = Literal["plausible_absent", "out_of_corpus"]
AuditStatus = Literal["pending", "accepted", "rejected", "edited"]

QID_PREFIX: dict[str, str] = {
    "paraphrase": "pa",
    "exact_term": "ex",
    "multi_hop": "mh",
    "unanswerable": "un",
}

_WS = re.compile(r"\s+")


def content_sha256(text: str) -> str:
    """Hash a chunk's stored content. The drift tripwire, in one place."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_qid(query_type: str, question: str, gold_keys: Iterable[tuple[str, int]]) -> str:
    """A content-derived id, stable across processes and machines.

    Derived rather than assigned because it is what makes a generation run
    resumable and an audit re-appliable: the same question over the same gold
    chunks is the same record no matter which run produced it, so appending to
    the silver file twice is idempotent and a verdict written in one session
    still finds its record in the next.

    Hashing the *question* (not just its position) is why editing a question
    during the audit must mint a new qid — an edited question is a different
    question, and silently keeping the old id would let a verdict about the
    original text follow the replacement.
    """
    normalized = _WS.sub(" ", question.strip().casefold())
    keys = "\x1e".join(f"{key}:{index}" for key, index in sorted(gold_keys))
    digest = hashlib.sha256(f"{query_type}\x1f{normalized}\x1f{keys}".encode()).hexdigest()
    return f"{QID_PREFIX.get(query_type, 'xx')}_{digest[:12]}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class GoldChunk(BaseModel):
    """One chunk that a question's answer is grounded in."""

    document_key: str
    chunk_index: int
    content_sha256: str
    snippet: str
    # "primary" chunks are the ones the question genuinely needs. "overlap"
    # entries exist because CHUNK_OVERLAP is 200 characters, so a verified
    # snippet very often appears verbatim in the neighbouring chunk too —
    # counting only the primary would score a retriever as wrong for returning
    # the adjacent chunk holding identical text, i.e. would bake a defect into
    # the ruler. Metrics count a hit on any entry; AllGold@k counts primaries.
    role: Literal["primary", "overlap"] = "primary"
    # Convenience only, and deliberately not part of the identity: a fresh
    # database mints new document UUIDs for the same corpus.
    document_id: str | None = None
    document_title: str | None = None
    section: str | None = None


class SeedChunk(BaseModel):
    """Which chunk an unanswerable question was written *near*.

    Kept because it is the difference between a pass/fail and a diagnosis. When
    an unanswerable question is answered rather than refused, the seed says
    whether retrieval surfaced the topically-nearest chunk and the model
    over-claimed from it (an F2 grounding failure) or whether retrieval returned
    something unrelated entirely (a retrieval failure). Those need different
    fixes and without the seed they look identical.
    """

    document_key: str
    chunk_index: int
    content_sha256: str


class Verification(BaseModel):
    """What the independent verifier saw before accepting an unanswerable."""

    top1_similarity: float | None = None
    retrieved_keys: list[tuple[str, int]] = Field(default_factory=list)
    judge_model: str | None = None
    judge_verdict: str | None = None


class Generation(BaseModel):
    """Provenance. ``served_model`` is the one that matters."""

    prompt_version: str
    requested_model: str
    # NOT decoration. LLMClient.chat always passes fallbacks=[fallback_model],
    # so a rate limit on OpenAI silently yields Claude-written records inside a
    # dataset described as gpt-4o-mini's. LLMResponse.model carries the truth.
    served_model: str
    temperature: float = 0.0
    generated_at: str = Field(default_factory=utc_now_iso)
    run_id: str = ""
    unit_id: str = ""
    sampler_seed: int | None = None


class Corpus(BaseModel):
    """The silent inputs the gold keys were computed under.

    The same three columns ``documents`` carries, for the same reason: a dataset
    generated under one chunker and measured under another is comparing two
    different systems and reporting the difference as a regression.
    """

    normalizer_version: str
    chunker_version: str
    embedding_model: str


class Gates(BaseModel):
    passed: bool = True
    warnings: list[str] = Field(default_factory=list)
    snippet_verified: bool = False
    verified_at: str | None = None


class Audit(BaseModel):
    status: AuditStatus = "pending"
    reviewer: str | None = None
    reviewed_at: str | None = None
    reason: str | None = None
    notes: str | None = None
    # An edit is recorded, not applied in place, so the curated file stays an
    # auditable trail rather than a rewrite of what the model actually produced.
    original_question: str | None = None
    original_answer: str | None = None


class EvalRecord(BaseModel):
    """One Q/A/context triple."""

    schema_version: int = SCHEMA_VERSION
    qid: str
    query_type: QueryType
    answerable: bool
    question: str
    answer: str | None = None
    gold: list[GoldChunk] = Field(default_factory=list)

    exact_term: str | None = None
    unanswerable_kind: UnanswerableKind | None = None
    seed: SeedChunk | None = None
    why_absent: str | None = None
    why_both_needed: str | None = None
    verification: Verification | None = None

    generation: Generation
    corpus: Corpus
    gates: Gates = Field(default_factory=Gates)
    audit: Audit = Field(default_factory=Audit)

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported eval record schema_version {value} "
                f"(this build reads {SCHEMA_VERSION}). Regenerate the dataset "
                f"rather than reading it partially."
            )
        return value

    @property
    def primary_gold_keys(self) -> list[tuple[str, int]]:
        return [(g.document_key, g.chunk_index) for g in self.gold if g.role == "primary"]

    @property
    def all_gold_keys(self) -> list[tuple[str, int]]:
        return [(g.document_key, g.chunk_index) for g in self.gold]


class RejectedRecord(BaseModel):
    """A candidate a gate threw away, kept with the reason it went.

    Not ``/dev/null``. The drop histogram is a finding in its own right — "18%
    of the generator's cited snippets were not verbatim" is a sentence worth
    publishing, and it is unrecoverable if the evidence is discarded.
    """

    drop_reason: str
    dropped_at: str = Field(default_factory=utc_now_iso)
    unit_id: str | None = None
    qid: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write records, one sorted-key JSON object per line.

    ``sort_keys`` so two runs producing the same records produce byte-identical
    files and ``git diff`` shows what actually changed rather than a key
    reordering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append one record and flush.

    Append-and-flush rather than collect-and-write so a run killed part-way
    leaves a valid file missing at most the in-flight item, instead of an empty
    one. Generation costs money; losing an hour of it to a SIGKILL is avoidable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
        handle.write("\n")
        handle.flush()


def read_jsonl(path: Path) -> Iterator[EvalRecord]:
    """Read a dataset file, validating every line.

    Yields lazily so a large file is never fully resident, and lets a caller
    stop early. A malformed line raises with its line number rather than being
    skipped — a dataset that silently loses records is one whose denominators
    are wrong.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield EvalRecord.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 — re-raised with position
                raise ValueError(f"{path}:{line_number} is not a valid eval record: {exc}") from exc


def read_unit_ids(*paths: Path) -> set[str]:
    """Every generation unit already attempted, across silver and rejected files.

    This is what makes a run resumable: a unit that produced records *or* was
    thrown away has been paid for, and re-attempting it spends money to arrive
    at the same place. Reading the rejected file too is the part that is easy to
    forget — without it, every unit whose output was gated out is retried on
    every run, forever.
    """
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                unit_id = payload.get("unit_id") or payload.get("generation", {}).get("unit_id")
                if unit_id:
                    seen.add(unit_id)
    return seen
