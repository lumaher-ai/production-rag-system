"""The human half of decision G1: render a review sheet, read verdicts back.

G1's call is "(b) seeded, then (a) hand-audited", and this is the (a). A
generated dataset that nobody has read is a plausible-looking instrument of
unknown accuracy, and every number measured against it inherits that unknown.

Three files, kept separate on purpose:

    dataset.silver.jsonl    machine-written, regenerable, never hand-edited
    audit/audit-<run>.md    the human edits ONLY this
    dataset.jsonl           curated; produced only by merging the two above

Because verdicts live in their own file and are keyed by ``qid``, regenerating
the silver set can never destroy review work — the merge simply reports any
verdict whose record no longer exists.
"""

import random
import re
from collections.abc import Sequence
from pathlib import Path

from production_rag.eval.corpus import CorpusIndex
from production_rag.eval.schema import EvalRecord, utc_now_iso

# 50 items, weighted by where LLM generation actually fails rather than
# uniformly. Multi-hop takes half its own stratum because its failure mode — a
# "multi-hop" question answerable from one chunk — is invisible to every
# automatic gate; a human is the only detector. Uniform sampling would spend the
# scarcest resource on paraphrase, which the gates already handle well.
AUDIT_QUOTAS: dict[str, int] = {
    "paraphrase": 12,
    "exact_term": 10,
    "multi_hop": 15,
    "unanswerable": 13,
}

VERDICT_KEYS = frozenset({"qid", "decision", "reason", "question", "answer"})
DECISIONS = frozenset({"accept", "reject", "edit"})

_VERDICT_BLOCK = re.compile(r"```verdict\s*\n(.*?)```", re.DOTALL)


class AuditError(ValueError):
    """The sheet cannot be applied. Raised rather than skipped, deliberately."""


# ─── Sampling ───


def select_for_audit(
    records: Sequence[EvalRecord],
    seed: int,
    quotas: dict[str, int] | None = None,
) -> list[EvalRecord]:
    """Choose which records a human should read.

    Flagged records first — anything a gate warned about is where a human's
    judgement is worth most, and letting the random draw skip them would make
    the audit sample look cleaner than the dataset it is supposed to describe.
    Then a seeded random fill to quota, so the sheet is reproducible.
    """
    quotas = quotas or AUDIT_QUOTAS
    rng = random.Random(seed)
    chosen: list[EvalRecord] = []

    for query_type, quota in quotas.items():
        stratum = sorted(
            (record for record in records if record.query_type == query_type),
            key=lambda record: record.qid,
        )
        flagged = [record for record in stratum if record.gates.warnings]
        clean = [record for record in stratum if not record.gates.warnings]
        rng.shuffle(clean)
        chosen.extend((flagged + clean)[:quota])

    return chosen


# ─── Rendering ───


def render_sheet(
    records: Sequence[EvalRecord],
    index: CorpusIndex,
    run_id: str,
) -> str:
    """The Markdown review sheet.

    Full chunk text is pulled live from the corpus rather than stored on the
    record, so the sheet always shows what the database currently holds — and
    when a chunk's hash no longer matches what the record was generated
    against, the sheet says so instead of quietly showing different text under
    a citation that no longer applies.
    """
    lines: list[str] = [
        f"# Eval audit sheet — run `{run_id}`",
        "",
        f"{len(records)} items. For each, set `decision:` to `accept`, `reject`, or `edit`.",
        "",
        "- **accept** — the question is answerable by someone who has never seen the chunk,",
        "  the answer is correct, and the cited snippet is the text that answers it.",
        "- **reject** — anything else. `reason:` is required.",
        "- **edit** — salvageable with a better question or answer. Fill `question:` and/or",
        "  `answer:`. An edited question gets a new id; the original is kept on the record.",
        "",
        "For multi-hop, the question that matters is: *does answering it genuinely need BOTH",
        "chunks?* If either alone would do, reject. For unanswerable, it is: *could anything in",
        "this corpus answer this?* If yes, reject.",
        "",
        "Leave nothing blank — `audit-apply` refuses a sheet with an unfilled decision rather",
        "than applying half of it.",
        "",
        "---",
        "",
    ]

    for position, record in enumerate(records, start=1):
        lines.append(
            f"## {position} / {len(records)} · `{record.qid}` · {record.query_type}"
        )
        if record.gates.warnings:
            lines.append(f"⚠️ warnings: {', '.join(record.gates.warnings)}")
        lines.append("")
        lines.append(f"**Q:** {record.question}")
        lines.append("")
        if record.answer:
            lines.append(f"**A:** {record.answer}")
            lines.append("")
        if record.why_both_needed:
            lines.append(f"**Why both chunks:** {record.why_both_needed}")
            lines.append("")
        if record.why_absent:
            lines.append(f"**Why unanswerable:** {record.why_absent}")
            lines.append("")
        if record.verification:
            similarity = record.verification.top1_similarity
            measured = f" · top-1 similarity {similarity:.3f}" if similarity is not None else ""
            lines.append(
                f"**Verifier ({record.verification.judge_model}):** "
                f"{record.verification.judge_verdict}{measured}"
            )
            lines.append("")

        primaries = [gold for gold in record.gold if gold.role == "primary"]
        for number, gold in enumerate(primaries, start=1):
            section = f" · § {gold.section}" if gold.section else ""
            lines.append(
                f"### Gold {number} — {gold.document_key} · chunk {gold.chunk_index}{section}"
            )
            lines.append("")
            lines.append(f"> **cited snippet:** {gold.snippet}")
            lines.append("")
            content = index.content(gold.document_key, gold.chunk_index)
            if content is None:
                lines.append("**⚠️ this chunk no longer exists in the corpus.**")
            elif index.sha256(gold.document_key, gold.chunk_index) != gold.content_sha256:
                lines.append(
                    "**⚠️ this chunk's content has CHANGED since generation — "
                    "the text below is not what the question was written against.**"
                )
            lines.append("")
            lines.append(
                f"<details><summary>full chunk text ({len(content or '')} chars)</summary>"
            )
            lines.append("")
            lines.append("```")
            lines.append(content or "(missing)")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if record.seed:
            seeded = index.content(record.seed.document_key, record.seed.chunk_index)
            lines.append(
                f"### Seeded near — {record.seed.document_key} · chunk {record.seed.chunk_index}"
            )
            lines.append("")
            lines.append("<details><summary>the chunk this question was written near</summary>")
            lines.append("")
            lines.append("```")
            lines.append(seeded or "(missing)")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("```verdict")
        lines.append(f"qid: {record.qid}")
        lines.append("decision:")
        lines.append("reason:")
        lines.append("question:")
        lines.append("answer:")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── Parsing ───


def parse_sheet(text: str) -> list[dict[str, str]]:
    """Read verdict blocks out of an edited sheet.

    Hand-rolled rather than YAML: no new dependency, and — more to the point — a
    YAML parser would silently accept a mistyped key as an extra mapping entry,
    which is exactly the failure this must catch. Continuation lines (a
    multi-line replacement answer) attach to the previous key.
    """
    verdicts: list[dict[str, str]] = []
    for block in _VERDICT_BLOCK.findall(text):
        fields: dict[str, str] = {}
        current: str | None = None
        for line in block.splitlines():
            match = re.match(r"^([a-z_]+):(.*)$", line)
            if match:
                key = match.group(1)
                if key not in VERDICT_KEYS:
                    raise AuditError(
                        f"Unknown key '{key}' in a verdict block. "
                        f"Known keys: {', '.join(sorted(VERDICT_KEYS))}."
                    )
                current = key
                fields[current] = match.group(2).strip()
            elif current and line.strip():
                fields[current] = (fields[current] + " " + line.strip()).strip()
        verdicts.append(fields)
    return verdicts


def validate_verdicts(
    verdicts: Sequence[dict[str, str]],
    known_qids: set[str],
) -> None:
    """Reject a sheet that cannot be applied cleanly. Abort, never skip.

    A half-applied audit is worse than a failed one: it silently produces a
    curated file whose contents depend on which items happened to parse. The
    reviewer is sitting right there and can fix a blank line in ten seconds.
    """
    problems: list[str] = []
    for position, verdict in enumerate(verdicts, start=1):
        qid = verdict.get("qid", "").strip()
        decision = verdict.get("decision", "").strip().lower()
        if not qid:
            problems.append(f"item {position}: no qid")
            continue
        if qid not in known_qids:
            problems.append(f"item {position} ({qid}): no such record in the silver file")
        if decision not in DECISIONS:
            problems.append(
                f"item {position} ({qid}): decision is "
                f"{decision or 'blank'!r}, expected one of {sorted(DECISIONS)}"
            )
            continue
        if decision == "reject" and not verdict.get("reason", "").strip():
            problems.append(f"item {position} ({qid}): reject needs a reason")
        if decision == "edit" and not (
            verdict.get("question", "").strip() or verdict.get("answer", "").strip()
        ):
            problems.append(f"item {position} ({qid}): edit needs a question or an answer")

    if problems:
        raise AuditError(
            "This sheet cannot be applied:\n  " + "\n  ".join(problems)
        )


# ─── Merging ───


def apply_verdicts(
    records: Sequence[EvalRecord],
    verdicts: Sequence[dict[str, str]],
    reviewer: str | None = None,
) -> tuple[list[EvalRecord], list[str]]:
    """Build the curated dataset. Returns ``(records, warnings)``.

    The curated set is **every silver record except the rejected ones**, each
    carrying its audit status. Not just the accepted 50: the human reviews a
    third of the file, so restricting the dataset to what was read would throw
    away two thirds of the instrument to gain nothing. What the audit buys is
    the **acceptance rate on a stratified sample** — a published number
    describing the whole file's quality — plus the removal of the specific bad
    items found.

    Idempotent: applying the same verdicts twice yields an identical file.
    """
    by_qid = {record.qid: record for record in records}
    decisions = {
        verdict["qid"].strip(): verdict
        for verdict in verdicts
        if verdict.get("qid", "").strip()
    }
    warnings = [
        f"verdict for {qid} has no matching record — it was probably regenerated; "
        f"the verdict is being ignored, not applied to a different question"
        for qid in decisions
        if qid not in by_qid
    ]

    curated: list[EvalRecord] = []
    for record in records:
        verdict = decisions.get(record.qid)
        if verdict is None:
            curated.append(record)
            continue

        decision = verdict["decision"].strip().lower()
        if decision == "reject":
            continue

        updated = record.model_copy(deep=True)
        updated.audit.reviewer = reviewer
        updated.audit.reviewed_at = utc_now_iso()
        updated.audit.notes = verdict.get("reason", "").strip() or None

        if decision == "accept":
            updated.audit.status = "accepted"
        else:
            new_question = verdict.get("question", "").strip()
            new_answer = verdict.get("answer", "").strip()
            updated.audit.status = "edited"
            if new_question and new_question != record.question:
                updated.audit.original_question = record.question
                updated.question = new_question
                # A changed question is a different question, so it gets a
                # different id. Keeping the old one would let a verdict about
                # the original text follow its replacement into the next audit.
                updated.qid = _requalify(updated)
            if new_answer and new_answer != record.answer:
                updated.audit.original_answer = record.answer
                updated.answer = new_answer

        curated.append(updated)

    return sorted(curated, key=lambda record: record.qid), warnings


def _requalify(record: EvalRecord) -> str:
    from production_rag.eval.schema import make_qid

    return make_qid(record.query_type, record.question, record.primary_gold_keys)


def acceptance_summary(
    curated: Sequence[EvalRecord],
    verdicts: Sequence[dict[str, str]],
) -> dict[str, object]:
    """The dataset's own published quality number.

    G1's Proof clause asks for inter-rater agreement on a sample. With one
    reviewer that is not available, and reporting a κ of 1.0 against oneself
    would be a lie dressed as a measurement. The honest substitute is the
    acceptance rate on the audited sample, reported alongside the fact that
    agreement was not measured — which is what ``cli stats`` prints.
    """
    counts = {"accept": 0, "reject": 0, "edit": 0}
    for verdict in verdicts:
        decision = verdict.get("decision", "").strip().lower()
        if decision in counts:
            counts[decision] += 1
    reviewed = sum(counts.values())
    return {
        "reviewed": reviewed,
        "accepted": counts["accept"],
        "edited": counts["edit"],
        "rejected": counts["reject"],
        "acceptance_rate": (counts["accept"] / reviewed) if reviewed else None,
        "usable_rate": ((counts["accept"] + counts["edit"]) / reviewed) if reviewed else None,
        "curated_size": len(curated),
        "inter_rater_agreement": "not measured (single reviewer)",
    }


def sheet_path(run_id: str, directory: Path) -> Path:
    return directory / f"audit-{run_id}.md"
