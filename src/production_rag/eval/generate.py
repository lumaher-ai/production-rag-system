"""Sample chunks, ask a model for questions, gate what comes back.

The shape of the run: every stratum over-generates, every candidate is checked
against the corpus it claims to cite, and what survives is trimmed to quota.
Over-generation is not waste — the gates discard roughly a third of raw items,
and a run that generated exactly 150 would deliver about 100.

Resumability is by ``unit_id``: a chunk (or pair, or seed) that has already been
sent to the model is not sent again, whether it produced records or was thrown
away entirely. Reading the *rejected* file for unit ids as well as the silver
one is the part that is easy to forget, and forgetting it means every unit whose
output was gated out gets retried on every run, forever.
"""

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from production_rag.config import Settings
from production_rag.eval import prompts
from production_rag.eval.corpus import CorpusIndex, read_out_of_corpus
from production_rag.eval.gates import (
    MAX_SNIPPET_CHARS,
    MIN_SNIPPET_CHARS,
    contains_exact_term,
    deduplicate,
    find_overlap_chunks,
    has_valid_length,
    is_self_referential,
    leaks_answer,
    recover_snippet,
)
from production_rag.eval.parsing import MalformedGenerationError, extract_items
from production_rag.eval.sampling import ChunkRef, sample_chunks, sample_pairs
from production_rag.eval.schema import (
    Corpus,
    EvalRecord,
    Gates,
    Generation,
    GoldChunk,
    RejectedRecord,
    SeedChunk,
    Verification,
    make_qid,
    utc_now_iso,
)
from production_rag.ingestion.normalize import NORMALIZER_VERSION
from production_rag.llm.client import LLMClient
from production_rag.logging_config import get_logger
from production_rag.services.document_service import CHUNKER_VERSION, DocumentService

logger = get_logger(__name__)

# The confirmed stratification (decision G1). Held as data rather than as
# literals scattered through the code so `stats` can assert against the same
# numbers the generator targeted.
QUOTAS: dict[str, int] = {
    "paraphrase": 50,
    "exact_term": 40,
    "multi_hop": 30,
    "unanswerable": 30,
}
# Of the 30 unanswerables, how many come from the hand-written off-domain file.
# The rest are plausible-absent, seeded from real chunks — the harder and more
# diagnostic flavour, hence the larger share.
OUT_OF_CORPUS_QUOTA = 10

# Over-generation factors. Gates discard 25-40% of raw items depending on
# stratum, and a short run costs another full pass to top up.
OVERSAMPLE = 1.6

# The default 1024 truncates a five-question reply carrying 400-character
# snippets, and a truncated reply is malformed JSON that gets blamed on the
# model rather than on the caller that cut it off.
MAX_TOKENS = {
    "single_chunk": 2048,
    "multi_hop": 1024,
    "unanswerable": 512,
    "verifier": 8,
}


@dataclass
class RunStats:
    """What the run did, including what it threw away and why."""

    run_id: str
    generated: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    served_models: dict[str, int] = field(default_factory=dict)
    verifier_models: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    cost_usd: float = 0.0
    units_skipped: int = 0

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def keep(self, query_type: str) -> None:
        self.generated[query_type] = self.generated.get(query_type, 0) + 1


def unit_id(kind: str, *parts: object) -> str:
    """A stable id for one unit of generation work.

    ``PROMPT_VERSION`` is in the hash on purpose: questions written under two
    different prompts are not the same population, so bumping the prompt should
    make a re-run regenerate rather than resume.
    """
    payload = "|".join([prompts.PROMPT_VERSION, kind, *(str(part) for part in parts)])
    return "u_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


class SilverGenerator:
    """Generates, gates, and hands back records. Writing them is the caller's job."""

    def __init__(
        self,
        llm: LLMClient,
        index: CorpusIndex,
        settings: Settings,
        run_id: str,
        seed: int,
        document_service: DocumentService | None = None,
        user_id: UUID | None = None,
        completed_units: set[str] | None = None,
        existing_counts: dict[str, int] | None = None,
    ) -> None:
        self._llm = llm
        self._index = index
        self._settings = settings
        self._run_id = run_id
        self._seed = seed
        # Checked *before* each unit is sent, not after its records come back.
        # Filtering afterwards would still pay for every call — the run would
        # look resumable in the output and cost full price every time.
        self._completed = completed_units or set()
        # What the silver file already holds, per stratum. The quota belongs to
        # the *dataset*, not to one run: without this, topping up a stratum that
        # came up short would re-fill every other stratum to full quota as well
        # and quietly produce a 300-question file.
        self._existing = existing_counts or {}
        # Only the unanswerable verification pass needs these; the rest of
        # generation is corpus-in, questions-out and must stay runnable without
        # a retrieval stack.
        self._documents = document_service
        self._user_id = user_id
        self._semaphore = asyncio.Semaphore(max(1, settings.eval_generation_concurrency))
        self.stats = RunStats(run_id=run_id)
        self.rejected: list[RejectedRecord] = []
        self._corpus = Corpus(
            normalizer_version=NORMALIZER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model=settings.embedding_model,
        )

    # ─── LLM plumbing ───

    async def _ask(
        self,
        system: str,
        user: str,
        schema: dict | None,
        max_tokens: int,
        model: str | None = None,
        role: str = "generator",
    ) -> tuple[str, str]:
        """One call. Returns ``(content, served_model)``.

        ``role`` keeps the verifier's calls out of the generator's served-model
        tally. Without it every run reports "N calls were served by a model
        other than gpt-4o-mini" and points at the verifier, which is *supposed*
        to be a different model — a warning that fires on correct behaviour is
        one people learn to ignore, including on the run where it is real.
        """
        async with self._semaphore:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": user}],
                model=model or self._settings.eval_generator_model,
                system=system,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format=schema,
            )
        self.stats.llm_calls += 1
        self.stats.cost_usd += response.cost_usd
        tally = self.stats.served_models if role == "generator" else self.stats.verifier_models
        tally[response.model] = tally.get(response.model, 0) + 1
        return response.content, response.model

    async def _ask_for_items(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
        unit: str,
    ) -> tuple[list[dict], str]:
        """Ask, parse, and on malformed JSON repair exactly once.

        A second failure quarantines this unit and the run continues. One bad
        chunk out of sixty must never abort a paid run.
        """
        content, served = await self._ask(system, user, schema, max_tokens)
        try:
            return extract_items(content), served
        except MalformedGenerationError:
            logger.warning("eval_generation_unparseable", unit_id=unit, attempt=1)

        repair_user = f"{prompts.REPAIR_INSTRUCTION}\n\n{user}\n\nYour previous reply:\n{content}"
        content, served = await self._ask(system, repair_user, schema, max_tokens)
        try:
            return extract_items(content), served
        except MalformedGenerationError as exc:
            logger.warning("eval_generation_unparseable", unit_id=unit, attempt=2, error=str(exc))
            self.stats.drop("unparseable_generation")
            self.rejected.append(
                RejectedRecord(
                    drop_reason="unparseable_generation",
                    unit_id=unit,
                    payload={"raw": content[:4000]},
                )
            )
            return [], served

    def _skip(self, unit: str) -> bool:
        """Whether this unit has already been paid for in an earlier run."""
        if unit not in self._completed:
            return False
        self.stats.units_skipped += 1
        return True

    def _provenance(self, served_model: str, unit: str) -> Generation:
        return Generation(
            prompt_version=prompts.PROMPT_VERSION,
            requested_model=self._settings.eval_generator_model,
            served_model=served_model,
            temperature=0.0,
            run_id=self._run_id,
            unit_id=unit,
            sampler_seed=self._seed,
        )

    # ─── Gold construction ───

    def _build_gold(self, ref: ChunkRef, snippet: str) -> tuple[GoldChunk | None, list[GoldChunk]]:
        """Verify a citation and expand it across the chunk overlap.

        Returns the primary gold entry (or ``None`` if the citation is not
        really in the chunk) plus any neighbouring chunks carrying the same
        span. See ``gates.find_overlap_chunks`` for why the neighbours matter.
        """
        content = self._index.content(ref.document_key, ref.chunk_index)
        if content is None:
            return None, []
        recovered = recover_snippet(snippet, content)
        if recovered is None:
            return None, []

        chunk = self._index.get(ref.document_key, ref.chunk_index)
        assert chunk is not None
        primary = GoldChunk(
            document_key=ref.document_key,
            chunk_index=ref.chunk_index,
            content_sha256=self._index.sha256(ref.document_key, ref.chunk_index) or "",
            snippet=recovered,
            role="primary",
            document_id=str(chunk.document_id),
            document_title=chunk.document_title,
            section=chunk.section,
        )
        overlaps = [
            GoldChunk(
                document_key=ref.document_key,
                chunk_index=index,
                content_sha256=self._index.sha256(ref.document_key, index) or "",
                snippet=recovered,
                role="overlap",
                document_id=str(chunk.document_id),
                document_title=chunk.document_title,
            )
            for index in find_overlap_chunks(
                recovered, self._index.neighbours(ref.document_key, ref.chunk_index)
            )
        ]
        return primary, overlaps

    def _shape_ok(self, question: str, snippet: str, unit: str, payload: dict) -> bool:
        """The question-shape gates, with the drop reason recorded."""
        for reason, failed in (
            ("question_length", not has_valid_length(question)),
            ("self_referential", is_self_referential(question)),
            ("answer_leaked_into_question", leaks_answer(question, snippet)),
        ):
            if failed:
                self.stats.drop(reason)
                self.rejected.append(
                    RejectedRecord(drop_reason=reason, unit_id=unit, payload=payload)
                )
                return False
        return True

    # ─── Stratum 1+2: paraphrase and exact-term ───

    async def _single_chunk(self, ref: ChunkRef) -> list[EvalRecord]:
        unit = unit_id("single", ref.document_key, ref.chunk_index)
        if self._skip(unit):
            return []
        chunk = self._index.get(ref.document_key, ref.chunk_index)
        if chunk is None:
            return []

        items, served = await self._ask_for_items(
            prompts.SINGLE_CHUNK_SYSTEM,
            prompts.SINGLE_CHUNK_USER.format(
                document_title=chunk.document_title,
                section=chunk.section or "(none)",
                content=chunk.content,
            ),
            prompts.SINGLE_CHUNK_SCHEMA,
            MAX_TOKENS["single_chunk"],
            unit,
        )

        records: list[EvalRecord] = []
        for item in items:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            query_type = item.get("query_type") or "paraphrase"
            term = (item.get("exact_term") or "").strip() or None
            payload = dict(item, unit=unit)

            if not question or not answer or not snippet:
                self.stats.drop("missing_field")
                self.rejected.append(
                    RejectedRecord(drop_reason="missing_field", unit_id=unit, payload=payload)
                )
                continue
            if not self._shape_ok(question, snippet, unit, payload):
                continue

            primary, overlaps = self._build_gold(ref, snippet)
            if primary is None:
                self.stats.drop("snippet_not_verbatim")
                self.rejected.append(
                    RejectedRecord(
                        drop_reason="snippet_not_verbatim", unit_id=unit, payload=payload
                    )
                )
                continue

            warnings: list[str] = []
            if not MIN_SNIPPET_CHARS <= len(primary.snippet) <= MAX_SNIPPET_CHARS:
                warnings.append("snippet_length")
            if overlaps:
                warnings.append("snippet_spans_chunk_overlap")

            # An exact-term question whose term is not literally in the chunk is
            # not dropped — it is almost always a serviceable paraphrase
            # question, and discarding it throws away a call that has been paid
            # for. Reclassifying keeps the exact-term slice honest (it is the
            # slice that must fail today, per E2/E5) without wasting the item.
            if query_type == "exact_term" and not (
                term and contains_exact_term(term, chunk.content)
            ):
                query_type = "paraphrase"
                term = None
                warnings.append("reclassified_from_exact_term")

            key = (primary.document_key, primary.chunk_index)
            records.append(
                EvalRecord(
                    qid=make_qid(query_type, question, [key]),
                    query_type=query_type,  # type: ignore[arg-type]
                    answerable=True,
                    question=question,
                    answer=answer,
                    gold=[primary, *overlaps],
                    exact_term=term,
                    generation=self._provenance(served, unit),
                    corpus=self._corpus,
                    gates=Gates(
                        warnings=warnings,
                        snippet_verified=True,
                        verified_at=utc_now_iso(),
                    ),
                )
            )
        return records

    # ─── Stratum 3: multi-hop ───

    async def _multi_hop(self, left: ChunkRef, right: ChunkRef) -> list[EvalRecord]:
        unit = unit_id(
            "multihop", left.document_key, left.chunk_index, right.chunk_index
        )
        if self._skip(unit):
            return []
        chunk_a = self._index.get(left.document_key, left.chunk_index)
        chunk_b = self._index.get(right.document_key, right.chunk_index)
        if chunk_a is None or chunk_b is None:
            return []

        items, served = await self._ask_for_items(
            prompts.MULTI_HOP_SYSTEM,
            prompts.MULTI_HOP_USER.format(
                document_title=chunk_a.document_title,
                content_a=chunk_a.content,
                content_b=chunk_b.content,
            ),
            prompts.MULTI_HOP_SCHEMA,
            MAX_TOKENS["multi_hop"],
            unit,
        )

        records: list[EvalRecord] = []
        for item in items:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            snippet_a = (item.get("snippet_a") or "").strip()
            snippet_b = (item.get("snippet_b") or "").strip()
            why = (item.get("why_both_needed") or "").strip()
            payload = dict(item, unit=unit)

            if not all((question, answer, snippet_a, snippet_b)):
                self.stats.drop("missing_field")
                self.rejected.append(
                    RejectedRecord(drop_reason="missing_field", unit_id=unit, payload=payload)
                )
                continue
            if not self._shape_ok(question, snippet_a + " " + snippet_b, unit, payload):
                continue

            primary_a, overlaps_a = self._build_gold(left, snippet_a)
            primary_b, overlaps_b = self._build_gold(right, snippet_b)
            if primary_a is None or primary_b is None:
                self.stats.drop("snippet_not_verbatim")
                self.rejected.append(
                    RejectedRecord(
                        drop_reason="snippet_not_verbatim", unit_id=unit, payload=payload
                    )
                )
                continue
            if primary_a.chunk_index == primary_b.chunk_index:
                # Both citations landed in the same chunk, so whatever the model
                # wrote, it is not a two-hop question.
                self.stats.drop("multi_hop_single_gold")
                self.rejected.append(
                    RejectedRecord(
                        drop_reason="multi_hop_single_gold", unit_id=unit, payload=payload
                    )
                )
                continue

            gold = [primary_a, primary_b, *overlaps_a, *overlaps_b]
            keys = [(primary_a.document_key, primary_a.chunk_index),
                    (primary_b.document_key, primary_b.chunk_index)]
            records.append(
                EvalRecord(
                    qid=make_qid("multi_hop", question, keys),
                    query_type="multi_hop",
                    answerable=True,
                    question=question,
                    answer=answer,
                    gold=gold,
                    why_both_needed=why or None,
                    generation=self._provenance(served, unit),
                    corpus=self._corpus,
                    gates=Gates(
                        warnings=["multi_hop_weak_link"] if not why else [],
                        snippet_verified=True,
                        verified_at=utc_now_iso(),
                    ),
                )
            )
        return records

    # ─── Stratum 4: unanswerable ───

    async def _unanswerable_from_chunk(self, ref: ChunkRef) -> list[EvalRecord]:
        unit = unit_id("unanswerable", ref.document_key, ref.chunk_index)
        if self._skip(unit):
            return []
        chunk = self._index.get(ref.document_key, ref.chunk_index)
        if chunk is None:
            return []

        items, served = await self._ask_for_items(
            prompts.UNANSWERABLE_SYSTEM,
            prompts.UNANSWERABLE_USER.format(
                document_title=chunk.document_title,
                section=chunk.section or "(none)",
                content=chunk.content,
            ),
            prompts.UNANSWERABLE_SCHEMA,
            MAX_TOKENS["unanswerable"],
            unit,
        )

        records: list[EvalRecord] = []
        for item in items:
            question = (item.get("question") or "").strip()
            payload = dict(item, unit=unit)
            if not question:
                self.stats.drop("missing_field")
                continue
            if not has_valid_length(question) or is_self_referential(question):
                reason = (
                    "question_length" if not has_valid_length(question) else "self_referential"
                )
                self.stats.drop(reason)
                self.rejected.append(
                    RejectedRecord(drop_reason=reason, unit_id=unit, payload=payload)
                )
                continue

            records.append(
                EvalRecord(
                    qid=make_qid("unanswerable", question, []),
                    query_type="unanswerable",
                    answerable=False,
                    question=question,
                    answer=None,
                    gold=[],
                    unanswerable_kind="plausible_absent",
                    seed=SeedChunk(
                        document_key=ref.document_key,
                        chunk_index=ref.chunk_index,
                        content_sha256=self._index.sha256(ref.document_key, ref.chunk_index) or "",
                    ),
                    why_absent=(item.get("why_absent") or "").strip() or None,
                    generation=self._provenance(served, unit),
                    corpus=self._corpus,
                    gates=Gates(snippet_verified=True, verified_at=utc_now_iso()),
                )
            )
        return records

    def _out_of_corpus_records(self) -> list[EvalRecord]:
        """The hand-written off-domain questions, as records. No LLM involved."""
        return [
            EvalRecord(
                qid=make_qid("unanswerable", question, []),
                query_type="unanswerable",
                answerable=False,
                question=question,
                answer=None,
                gold=[],
                unanswerable_kind="out_of_corpus",
                generation=Generation(
                    prompt_version=prompts.PROMPT_VERSION,
                    requested_model="hand-written",
                    served_model="hand-written",
                    run_id=self._run_id,
                    unit_id=unit_id("outofcorpus", question),
                    sampler_seed=self._seed,
                ),
                corpus=self._corpus,
                gates=Gates(snippet_verified=True, verified_at=utc_now_iso()),
            )
            for question in read_out_of_corpus()
            if not self._skip(unit_id("outofcorpus", question))
        ]

    async def _retrieve_for(self, record: EvalRecord) -> list:
        """Retrieval for one negative candidate. **Called serially, never fanned out.**

        ``AsyncSession`` is not safe for concurrent use: two coroutines awaiting
        on one session raise "this session is already flushing" or "concurrent
        operations are not permitted", and inside a ``gather(...,
        return_exceptions=True)`` that surfaces as a pile of quarantined
        candidates rather than as an error anybody can read. The first live run
        of this file lost 26 of 32 negatives to exactly that.

        Retrieval is a local query plus one embedding call, so serializing it
        costs seconds; the judge calls that follow are the slow part and those
        still run concurrently.
        """
        if self._documents is None or self._user_id is None:
            return []
        return await self._documents.retrieve(
            question=record.question, user_id=self._user_id, top_k=5, use_cache=False
        )

    async def verify_unanswerable(  # noqa: D401
        self, record: EvalRecord, scored: list | None = None
    ) -> bool:
        """Whether the corpus really cannot answer this question.

        The generator sees one chunk; the corpus has hundreds. So the negative
        it writes may well be answered three documents away, and a negative the
        corpus can answer converts correct behaviour into a scored failure —
        permanently, and invisibly, because nothing downstream can tell the
        difference.

        Real retrieval, then a **different model** decides. A similarity
        threshold alone would not do: a genuinely unanswerable question about a
        chunk's topic has *high* similarity to that chunk by construction, so
        the check has to be semantic. Using the fallback model rather than the
        generator is the same self-judging argument G3 makes about the answer
        judge, applied one phase early — and it means that judge harness is
        already built and exercised by the time G3 needs it.

        Fails **closed**: an unparseable verdict discards the candidate. A bad
        negative poisons a metric forever; a discarded one costs a tenth of a
        cent to replace.
        """
        if self._documents is None or self._user_id is None:
            return True

        # Retrieval is passed in by the caller, which does it serially — see
        # ``_retrieve_for``. Doing it here would put a shared AsyncSession inside
        # a fan-out.
        if scored is None:
            scored = await self._retrieve_for(record)
        passages = "\n\n---\n\n".join(item.chunk.content for item in scored)
        content, _ = await self._ask(
            prompts.VERIFIER_SYSTEM,
            prompts.VERIFIER_USER.format(question=record.question, passages=passages),
            None,
            MAX_TOKENS["verifier"],
            model=self._settings.eval_verifier_model,
            role="verifier",
        )
        verdict = content.strip().upper()

        record.verification = Verification(
            top1_similarity=scored[0].score if scored else None,
            retrieved_keys=[
                (self._index.key_for(item.chunk.document_id) or "?", item.chunk.chunk_index)
                for item in scored
            ],
            judge_model=self._settings.eval_verifier_model,
            judge_verdict=verdict[:8],
        )
        if scored and scored[0].score > 0.80:
            record.gates.warnings.append("unanswerable_high_similarity")

        return verdict.startswith("NO")

    # ─── The run ───

    async def generate(self, refs: Sequence[ChunkRef]) -> list[EvalRecord]:
        """Generate every stratum concurrently, then gate, dedup and trim."""
        single_target = int(
            (QUOTAS["paraphrase"] + QUOTAS["exact_term"]) / 3.5 * OVERSAMPLE
        )
        pair_target = int(QUOTAS["multi_hop"] / 1.2 * OVERSAMPLE)
        negative_target = int((QUOTAS["unanswerable"] - OUT_OF_CORPUS_QUOTA) / 1.5 * OVERSAMPLE)

        single = sample_chunks(refs, single_target, self._seed)
        pairs = sample_pairs(refs, pair_target, self._seed + 1)
        # Offset the seed so the negative slice is not written from the very
        # chunks the positive slice already mined — questions and their own
        # near-misses drawn from one passage correlate the two strata.
        negatives = sample_chunks(refs, negative_target, self._seed + 2)

        tasks = [
            *(self._single_chunk(ref) for ref in single),
            *(self._multi_hop(left, right) for left, right in pairs),
            *(self._unanswerable_from_chunk(ref) for ref in negatives),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[EvalRecord] = list(self._out_of_corpus_records())
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("eval_generation_unit_failed", error=str(result))
                self.stats.drop("unit_exception")
                continue
            candidates.extend(result)

        return await self._finalize(candidates)

    async def _finalize(self, candidates: list[EvalRecord]) -> list[EvalRecord]:
        by_qid: dict[str, EvalRecord] = {}
        for record in candidates:
            by_qid.setdefault(record.qid, record)

        survivors, duplicates = deduplicate(
            [(qid, record.question) for qid, record in by_qid.items()]
        )
        for dropped_qid, kept_qid in duplicates.items():
            self.stats.drop("near_duplicate")
            self.rejected.append(
                RejectedRecord(
                    drop_reason=f"near_duplicate_of:{kept_qid}",
                    qid=dropped_qid,
                    payload={"question": by_qid[dropped_qid].question},
                )
            )

        kept = [by_qid[qid] for qid in survivors]

        # Verify the negatives only after dedup: verification costs a retrieval
        # plus a judge call each, and paying for a candidate that a free string
        # comparison was about to delete is pure waste.
        verified: list[EvalRecord] = []
        negatives = [record for record in kept if not record.answerable]
        # Retrieval first and serially — one AsyncSession cannot serve a fan-out
        # — then the judge calls concurrently, which is where the latency is.
        retrievals: list[list] = []
        for record in negatives:
            try:
                retrievals.append(await self._retrieve_for(record))
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                logger.warning(
                    "eval_unanswerable_retrieval_failed", qid=record.qid, error=str(exc)
                )
                retrievals.append([])
        checks = await asyncio.gather(
            *(
                self.verify_unanswerable(record, scored)
                for record, scored in zip(negatives, retrievals, strict=True)
            ),
            return_exceptions=True,
        )
        for record, outcome in zip(negatives, checks, strict=True):
            if outcome is True:
                verified.append(record)
                continue
            if isinstance(outcome, BaseException):
                reason = "unanswerable_verification_failed"
                logger.warning(
                    "eval_unanswerable_verification_failed",
                    qid=record.qid,
                    error=str(outcome),
                )
            else:
                reason = "unanswerable_is_actually_answerable"
            self.stats.drop(reason)
            self.rejected.append(
                RejectedRecord(
                    drop_reason=reason, qid=record.qid, payload={"question": record.question}
                )
            )

        pool = [record for record in kept if record.answerable] + verified
        return self._trim_to_quota(pool)

    def _trim_to_quota(self, records: list[EvalRecord]) -> list[EvalRecord]:
        """Cut this run's output to what the *dataset* still needs.

        The quota is subtracted against what the silver file already holds, so
        topping up one short stratum cannot refill the three that were already
        complete.
        """
        remaining = {
            query_type: max(0, target - self._existing.get(query_type, 0))
            for query_type, target in QUOTAS.items()
        }
        kept, dropped = select_within_quota(records, remaining)
        for record in dropped:
            self.stats.drop(f"over_quota:{record.query_type}")
        for query_type, target in QUOTAS.items():
            have = self._existing.get(query_type, 0) + sum(
                1 for record in kept if record.query_type == query_type
            )
            if have < target:
                logger.warning(
                    "eval_stratum_under_quota",
                    query_type=query_type,
                    have=have,
                    want=target,
                    hint="re-run with a different --seed to sample more chunks",
                )
        for record in kept:
            self.stats.keep(record.query_type)
        return kept


def select_within_quota(
    records: Sequence[EvalRecord],
    quotas: dict[str, int],
) -> tuple[list[EvalRecord], list[EvalRecord]]:
    """Cut each stratum to its quota, deterministically. Returns ``(kept, dropped)``.

    Sorted rather than sampled, so the same pool always yields the same dataset —
    the trim has to be a function of its input, or the file stops being
    reproducible at the very last step after everything upstream went to such
    trouble to be.

    Records carrying gate warnings are kept **first**. They are the ones most
    worth a human's attention, and dropping them would leave the audit sample
    looking cleaner than the dataset actually is.
    """
    kept: list[EvalRecord] = []
    dropped: list[EvalRecord] = []
    for query_type, quota in quotas.items():
        stratum = sorted(
            (record for record in records if record.query_type == query_type),
            key=lambda record: (not record.gates.warnings, record.qid),
        )
        kept.extend(stratum[:quota])
        dropped.extend(stratum[quota:])
    return sorted(kept, key=lambda record: record.qid), dropped
