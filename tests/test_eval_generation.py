"""Generation, gating and quota-trimming — with a fake model, never a real one."""

import json
from uuid import uuid4

from production_rag.config import Settings
from production_rag.eval.corpus import CorpusIndex
from production_rag.eval.generate import QUOTAS, SilverGenerator, unit_id
from production_rag.eval.sampling import ChunkRef
from production_rag.models.document import DocumentChunk

from ._eval import FakeLLMClient, make_record

DOCUMENT_ID = uuid4()
CHUNK_ZERO = (
    "Iterative scan re-scans until the limit is satisfied. "
    "The hnsw.ef_search setting widens each pass."
)
CHUNK_TAIL = "SET LOCAL scopes both settings to the surrounding transaction only."


def build_index() -> CorpusIndex:
    contents = {0: CHUNK_ZERO, 1: "An unrelated middle chunk about loaders.", 5: CHUNK_TAIL}
    chunks = [
        DocumentChunk(
            document_id=DOCUMENT_ID,
            owner_id=uuid4(),
            document_title="doc-a",
            chunk_index=index,
            content=content,
            token_count=200,
            embedding=[0.0],
        )
        for index, content in contents.items()
    ]
    return CorpusIndex(chunks, {DOCUMENT_ID: "doc-a.md"})


def build_generator(llm: FakeLLMClient) -> SilverGenerator:
    return SilverGenerator(
        llm=llm,  # type: ignore[arg-type]
        index=build_index(),
        settings=Settings(eval_generation_concurrency=2),
        run_id="test",
        seed=1,
    )


def ref(chunk_index: int = 0) -> ChunkRef:
    return ChunkRef("doc-a.md", chunk_index, 200, "doc-a")


def reply(*items: dict) -> str:
    return json.dumps({"items": list(items)})


# ─── The happy path ───


async def test_a_verbatim_citation_produces_a_record_with_its_gold_chunk():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "query_type": "paraphrase",
                    "question": "How does the index avoid returning too few rows?",
                    "answer": "It re-scans until the limit is met.",
                    "snippet": "Iterative scan re-scans until the limit is satisfied.",
                    "exact_term": None,
                }
            )
        ]
    )
    records = await build_generator(llm)._single_chunk(ref())
    assert len(records) == 1
    assert records[0].gold[0].chunk_index == 0
    assert records[0].gates.snippet_verified


async def test_a_snippet_the_model_invented_is_dropped_with_its_reason():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "query_type": "paraphrase",
                    "question": "How does the index avoid returning too few rows?",
                    "answer": "It doubles the candidate list.",
                    "snippet": "Iterative scan doubles the candidate list each pass.",
                    "exact_term": None,
                }
            )
        ]
    )
    generator = build_generator(llm)
    assert await generator._single_chunk(ref()) == []
    assert generator.stats.dropped["snippet_not_verbatim"] == 1
    assert generator.rejected[0].drop_reason == "snippet_not_verbatim"


async def test_a_self_referential_question_is_dropped():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "query_type": "paraphrase",
                    "question": "According to the document, what does iterative scan do?",
                    "answer": "It re-scans.",
                    "snippet": "Iterative scan re-scans until the limit is satisfied.",
                    "exact_term": None,
                }
            )
        ]
    )
    generator = build_generator(llm)
    assert await generator._single_chunk(ref()) == []
    assert generator.stats.dropped["self_referential"] == 1


async def test_an_exact_term_question_with_no_rare_shared_token_becomes_a_paraphrase():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "query_type": "exact_term",
                    "question": "What does the invented_setting_name option control?",
                    "answer": "It widens each pass.",
                    "snippet": "The hnsw.ef_search setting widens each pass.",
                    "exact_term": "invented_setting_name",
                }
            )
        ]
    )
    records = await build_generator(llm)._single_chunk(ref())
    # Reclassified rather than discarded: it is a serviceable paraphrase and the
    # call has already been paid for.
    assert records[0].query_type == "paraphrase"
    assert records[0].exact_term is None
    assert "reclassified_from_exact_term" in records[0].gates.warnings


async def test_an_exact_term_present_in_the_chunk_is_kept_as_exact_term():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "query_type": "exact_term",
                    "question": "What does the hnsw.ef_search setting control?",
                    "answer": "It widens each pass.",
                    "snippet": "The hnsw.ef_search setting widens each pass.",
                    "exact_term": "hnsw.ef_search",
                }
            )
        ]
    )
    records = await build_generator(llm)._single_chunk(ref())
    assert records[0].query_type == "exact_term"
    assert records[0].exact_term == "hnsw.ef_search"


# ─── Malformed output ───


async def test_malformed_json_is_repaired_once_and_then_kept():
    llm = FakeLLMClient(
        replies=[
            "I'm sorry, I can't do that.",
            reply(
                {
                    "query_type": "paraphrase",
                    "question": "How does the index avoid returning too few rows?",
                    "answer": "It re-scans.",
                    "snippet": "Iterative scan re-scans until the limit is satisfied.",
                    "exact_term": None,
                }
            ),
        ]
    )
    generator = build_generator(llm)
    records = await generator._single_chunk(ref())
    assert len(records) == 1
    assert generator.stats.llm_calls == 2


async def test_a_unit_that_fails_twice_is_quarantined_and_the_run_continues():
    llm = FakeLLMClient(replies=["not json", "still not json"])
    generator = build_generator(llm)
    assert await generator._single_chunk(ref()) == []
    assert generator.stats.dropped["unparseable_generation"] == 1
    assert generator.rejected[0].drop_reason == "unparseable_generation"


# ─── Multi-hop ───


async def test_a_multi_hop_record_carries_two_distinct_primary_gold_chunks():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "question": "Does the scan setting apply beyond its own transaction?",
                    "answer": "No — it is scoped to the surrounding transaction.",
                    "snippet_a": "The hnsw.ef_search setting widens each pass.",
                    "snippet_b": "SET LOCAL scopes both settings",
                    "why_both_needed": "A names the setting; B gives its scope.",
                }
            )
        ]
    )
    records = await build_generator(llm)._multi_hop(ref(0), ref(5))
    assert len(records[0].primary_gold_keys) == 2
    assert records[0].query_type == "multi_hop"


async def test_a_multi_hop_record_whose_citation_is_invented_is_dropped():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "question": "Does the scan setting apply beyond its own transaction?",
                    "answer": "No.",
                    "snippet_a": "The hnsw.ef_search setting widens each pass.",
                    "snippet_b": "A sentence that appears in neither chunk at all.",
                    "why_both_needed": "A names it; B scopes it.",
                }
            )
        ]
    )
    generator = build_generator(llm)
    assert await generator._multi_hop(ref(0), ref(5)) == []
    assert generator.stats.dropped["snippet_not_verbatim"] == 1


async def test_a_multi_hop_with_no_why_both_needed_is_flagged_for_the_human():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "question": "Does the scan setting apply beyond its own transaction?",
                    "answer": "No.",
                    "snippet_a": "The hnsw.ef_search setting widens each pass.",
                    "snippet_b": "SET LOCAL scopes both settings",
                    "why_both_needed": "",
                }
            )
        ]
    )
    records = await build_generator(llm)._multi_hop(ref(0), ref(5))
    assert "multi_hop_weak_link" in records[0].gates.warnings


# ─── Unanswerable ───


async def test_an_unanswerable_record_carries_no_gold_chunks_but_keeps_its_seed():
    llm = FakeLLMClient(
        replies=[
            reply(
                {
                    "question": "What is the measured p99 latency of an iterative scan?",
                    "why_absent": "The passage never states a latency figure.",
                }
            )
        ]
    )
    records = await build_generator(llm)._unanswerable_from_chunk(ref())
    assert records[0].gold == []
    assert records[0].answerable is False
    assert records[0].answer is None
    assert records[0].unanswerable_kind == "plausible_absent"
    # The seed is what distinguishes "refused despite retrieving the right
    # chunk" from "retrieved nothing" when this question is later scored.
    assert records[0].seed is not None
    assert records[0].seed.chunk_index == 0


async def test_verification_is_skipped_without_a_retrieval_stack_rather_than_crashing():
    generator = build_generator(FakeLLMClient())
    assert await generator.verify_unanswerable(make_record(query_type="unanswerable")) is True


# ─── Quota trimming ───


def test_trimming_cuts_each_stratum_to_its_quota():
    generator = build_generator(FakeLLMClient())
    records = [
        make_record("paraphrase", f"Paraphrase question number {i} about the fox?")
        for i in range(QUOTAS["paraphrase"] + 25)
    ]
    trimmed = generator._trim_to_quota(records)
    assert len(trimmed) == QUOTAS["paraphrase"]
    assert generator.stats.generated["paraphrase"] == QUOTAS["paraphrase"]


def test_trimming_is_deterministic_rather_than_sampled():
    records = [
        make_record("paraphrase", f"Paraphrase question number {i} about the fox?")
        for i in range(QUOTAS["paraphrase"] + 25)
    ]
    first = build_generator(FakeLLMClient())._trim_to_quota(records)
    second = build_generator(FakeLLMClient())._trim_to_quota(list(reversed(records)))
    assert [record.qid for record in first] == [record.qid for record in second]


def test_flagged_records_survive_the_trim_ahead_of_clean_ones():
    flagged = make_record(
        "paraphrase", "A flagged question about the fox?", warnings=["snippet_length"]
    )
    clean = [
        make_record("paraphrase", f"A clean question number {i} about the dog?")
        for i in range(QUOTAS["paraphrase"] + 5)
    ]
    trimmed = build_generator(FakeLLMClient())._trim_to_quota([*clean, flagged])
    # Dropping the flagged ones would leave the audit sample looking cleaner
    # than the dataset actually is.
    assert flagged.qid in {record.qid for record in trimmed}


def test_a_stratum_below_quota_is_kept_whole_rather_than_padded():
    records = [make_record("multi_hop", "A multi hop question about two chunks?")]
    trimmed = build_generator(FakeLLMClient())._trim_to_quota(records)
    assert len(trimmed) == 1


# ─── Resumability ───


async def test_a_chunk_already_generated_is_never_sent_to_the_model_again():
    llm = FakeLLMClient()
    generator = SilverGenerator(
        llm=llm,  # type: ignore[arg-type]
        index=build_index(),
        settings=Settings(),
        run_id="test",
        seed=1,
        completed_units={unit_id("single", "doc-a.md", 0)},
    )
    assert await generator._single_chunk(ref(0)) == []
    # Filtering after the fact would still have paid for the call — the run
    # would look resumable in the output and cost full price every time.
    assert llm.calls == []
    assert generator.stats.llm_calls == 0
    assert generator.stats.units_skipped == 1


async def test_a_chunk_not_yet_generated_is_still_sent():
    llm = FakeLLMClient()
    generator = SilverGenerator(
        llm=llm,  # type: ignore[arg-type]
        index=build_index(),
        settings=Settings(),
        run_id="test",
        seed=1,
        completed_units={unit_id("single", "doc-a.md", 99)},
    )
    await generator._single_chunk(ref(0))
    assert generator.stats.llm_calls == 1


def test_a_unit_id_is_stable_for_the_same_chunk():
    assert unit_id("single", "doc-a.md", 3) == unit_id("single", "doc-a.md", 3)


def test_a_unit_id_differs_between_strata_over_the_same_chunk():
    assert unit_id("single", "doc-a.md", 3) != unit_id("unanswerable", "doc-a.md", 3)


async def test_generation_records_the_model_that_actually_served_the_call():
    llm = FakeLLMClient(
        model="claude-haiku-4-5-20251001",
        replies=[
            reply(
                {
                    "query_type": "paraphrase",
                    "question": "How does the index avoid returning too few rows?",
                    "answer": "It re-scans.",
                    "snippet": "Iterative scan re-scans until the limit is satisfied.",
                    "exact_term": None,
                }
            )
        ],
    )
    records = await build_generator(llm)._single_chunk(ref())
    # A rate limit on OpenAI silently serves from the fallback; a dataset that
    # only recorded the *requested* model would misdescribe itself.
    assert records[0].generation.requested_model == "gpt-4o-mini"
    assert records[0].generation.served_model == "claude-haiku-4-5-20251001"


def test_the_quota_counts_what_the_dataset_already_holds_not_just_this_run():
    generator = SilverGenerator(
        llm=FakeLLMClient(),  # type: ignore[arg-type]
        index=build_index(),
        settings=Settings(),
        run_id="test",
        seed=1,
        existing_counts={"paraphrase": QUOTAS["paraphrase"] - 3},
    )
    records = [
        make_record("paraphrase", f"Paraphrase question number {i} about the fox?")
        for i in range(20)
    ]
    # Topping up a short stratum must not refill one that is already full —
    # otherwise a second run quietly doubles the file.
    assert len(generator._trim_to_quota(records)) == 3


def test_a_stratum_already_at_quota_accepts_nothing_further():
    generator = SilverGenerator(
        llm=FakeLLMClient(),  # type: ignore[arg-type]
        index=build_index(),
        settings=Settings(),
        run_id="test",
        seed=1,
        existing_counts={"paraphrase": QUOTAS["paraphrase"]},
    )
    records = [
        make_record("paraphrase", f"Paraphrase question number {i} about the fox?")
        for i in range(5)
    ]
    assert generator._trim_to_quota(records) == []
