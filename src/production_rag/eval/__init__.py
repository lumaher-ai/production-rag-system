"""The evaluation instrument (decision G1).

Nothing in Parts A–F of ``docs/rag-production-decisions.md`` can be *decided*
without this package: every "Proof" line in that document is a number that
requires a dataset to produce, and until one exists the chunk size, the
embedding model, and the index parameters are defaults rather than defended
choices.

This package builds the dataset — it does not yet measure anything with it. The
retrieval metrics (G2) and the generation judge (G3) consume the JSONL this
writes; ``metrics.recall_at_k`` is here only because the trivial-baseline check
that G1's own Proof clause demands cannot run without it.

The pipeline, in order:

    seed-corpus   ingest eval/corpus/*.md under a dedicated eval user
    generate      sample chunks -> LLM -> gate -> eval/dataset.silver.jsonl
    baseline      prove a trivial retriever does NOT score on it
    audit-sheet   render 50 stratified items as Markdown for a human
    audit-apply   merge the human's verdicts -> eval/dataset.jsonl

The silver file is machine-written and regenerable; the curated file is only
ever produced from human verdicts. They are separate files precisely so that
re-running generation can never destroy review work.
"""
