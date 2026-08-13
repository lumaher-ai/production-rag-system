---
name: pr
description: Write a pull request description as an architectural report — the decisions, the criterion behind each one, and the trade-off accepted — then open the PR. Use whenever a branch is ready to merge, or when asked for a PR, a PR description, a branch write-up, or a rewrite of an existing PR body.
disable-model-invocation: true
---

`$ARGUMENTS` may name a branch, an existing PR number, or nothing — default to the current branch
against `main`.

Two people read this description: a senior engineer deciding whether to approve, and someone deciding
whether the time was well spent. Neither wants a changelog. Both want to know what was chosen, on
what criterion, and what it cost.

## Describe the system, not the diff

The reviewer already has the diff — GitHub is showing it on the same screen. What they cannot
recover from it is why the code took *this* shape rather than the two other shapes it could have
taken. That gap is the only thing a description can fill, so spend the entire budget there.

In practice: no file paths, no module or function names, no "renamed X to Y", no per-directory
walkthrough. A useful test — if a sentence would stop making sense after a pure refactor that changed
no behavior, it was describing the diff.

**Do not imitate the earliest PRs in this repo.** #1 and #2 are file-by-file walkthroughs; they are
the style this skill exists to replace. #4's "Key design decisions" section and #5 are the direction.
Read a recent one for register, not for structure.

## Finding the decisions

Read what actually landed, and read `docs/rag-production-decisions.md` where the work touches a
lettered decision — that document already holds the reasoning, and a PR that contradicts it is
usually the PR that's wrong. Look for the points where the branch could plausibly have gone another
way.

The test for whether something is a decision: **can you name the alternative?** If not, it's a
description, and it belongs in the opening paragraph as a clause, or nowhere. "Refactored for
maintainability" fails. "Chose X over Y on criterion Z, accepting cost C" passes.

Most branches hold two to six real decisions. Ten means some are descriptions. One means the obvious
choices went unexamined — an obvious choice is still a choice, and the criterion that made it obvious
is worth a sentence.

## The shape of a decision bullet

Four things, in whatever order reads best:

- **The choice**, as a claim rather than an activity — "Blocking on every PR", not "Added a blocking
  check".
- **The alternative** that was genuinely available.
- **The criterion** that decided between them — what you'd say out loud in a design review.
- **The cost** accepted. Every real decision has one; a bullet without a cost is advertising.

Weak:

> **Async ingestion.** Moved ingestion to a background worker for better performance and scalability.

Strong:

> **Uploads return immediately and finish out of band.** Doing the work inside the request is simpler
> and needs no queue, but it ties the user's timeout to the size of their document and loses
> everything on a crash. Cost: a job now has a status a caller has to poll, and "uploaded" no longer
> means "searchable".

The second one tells a reviewer what to argue with. The first is unfalsifiable.

## Open with what changed for someone using the system

One short paragraph, before the decisions: what the system can now do that it couldn't, or what risk
it no longer carries — in the vocabulary of the product, not the implementation. If the branch is
pure infrastructure, say what it protects and who from. A reader who stops after this paragraph
should still know whether the branch mattered.

## State the boundary you didn't cross

The strongest section in a senior PR is usually the one saying what this deliberately does *not* do.
It stops a reviewer assuming a neighbouring problem got solved, and it names the follow-up before
someone else has to discover it.

Include it when a reader would otherwise reasonably assume more was covered — a related decision left
open, a gate that only half exists, a capability the title implies but doesn't deliver. If there is
genuinely no such boundary, leave the section out rather than padding it.

## Hold the line this repo holds everywhere else

`docs/rag-production-decisions.md` separates *decided*, *implemented*, and *proven*, and the
description must not blur what that document keeps apart. "Passes the full gate" when the gate has
only ever run locally is the exact failure mode — say "passes locally; this PR is the first real
run." Verify anything the description asserts as fact; if you can't, mark it unverified in the text
rather than dropping it.

Link decision ids (`A2`, `D3`, `G5`) where the work closes, moves, or is blocked by one. If a
decision's state actually changed, that's `/decision`'s job and the README table moves with it —
mention it as follow-up rather than doing it here.

Keep the whole thing shorter than the diff deserves. Bullets of one to three sentences; PR #5 is
about the right length.

## Opening it

Title follows the repo's commit convention — `feat:` / `fix:` / `docs:`, lowercase, and the subject
states the *behavior*, not the work. "feat: every PR now runs lint, types, and the full test suite",
not "feat: add CI workflow".

Show the proposed title and body and wait for approval before creating anything.

On approval, write the body to a file and pass `--body-file` — inline `--body` mangles backticks and
`$` in a shell. Base is `main` unless told otherwise.

**Check whether the branch is pushed first.** If it isn't, stop and ask — pushing writes state to the
remote, and per `CLAUDE.local.md` no git command that writes state runs without approval. Same for a
branch that has diverged from its upstream.

If `$ARGUMENTS` named an existing PR, edit that one rather than opening a second.

Afterwards, run the ownership check from `CLAUDE.local.md` — writing the body file is a tool write
like any other.
