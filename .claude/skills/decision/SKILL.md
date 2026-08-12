---
name: decision
description: Record or close an engineering decision in docs/rag-production-decisions.md, in that document's established format, and keep the README status table in sync. Use with the decision id or a description, e.g. /decision E5 or /decision we picked a cross-encoder reranker.
disable-model-invocation: true
---

Target: `$ARGUMENTS` — either a decision id (`A3`, `D3`, `E4`, `G1`, …) or a description of the
choice that was just made. If it's a description, find the decision it belongs to before writing.

## The format is not yours to invent

Read `docs/rag-production-decisions.md` first — at minimum the "How to read this document" section
and two or three neighbouring entries in the same Part. Every entry is `### <Id>. <Question as a
question?>` followed by these bolded fields:

| Field | Meaning |
|---|---|
| **State** | ✅ decided & implemented · ⚠️ implicitly decided (default, never justified/measured) · ❌ open |
| **Now** | What the repo actually does today — verified against source, not against the roadmap |
| **Options** | The realistic choices, lettered (a)/(b)/(c) |
| **Call** | The recommended decision for *this* system, with the reason |
| **Proof** | The measurement that would let you defend the call in an interview or a postmortem |

When an entry changes state, entries in this document keep their history rather than overwriting it:
a **Was:** field records what it used to do and why that was wrong, and **Call** notes when the
original call was reversed and by what. Follow that pattern — do not silently rewrite an old entry.

## Verify before you write

**Now** must be checked against the source, not asserted. Read the code the entry describes. If the
implementation doesn't match what you're about to claim, say so and stop — a wrong **Now** is the
one failure mode this document is built to prevent.

Hold the ⚠️ / ✅ line honestly:

- ✅ means decided **and implemented**.
- ⚠️ means a default is running with no evidence behind it.
- Implemented but unproven is **not** ✅ with a caveat — D3 is the reference case: the mitigation is
  applied and asserted in tests, and the entry still says plainly that nothing has demonstrated it
  works at scale.

## Then sync the README

`README.md` carries a "Status: what works today" table and a "Known limitations" list keyed to these
decision ids. A state change here almost always means a row or bullet there. Update both, or say
explicitly why the README is unaffected.

Show the proposed diff for both files and wait for approval before writing.
