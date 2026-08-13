---
name: journal
description: Append a dated entry to JOURNAL.md summarizing what was built and learned, in the existing Day-N format. Use when asked to write up the day, log the session, or update the journal.
disable-model-invocation: true
---

Append to `JOURNAL.md`. Read the last two or three entries first — the format has drifted over time
and the newest entries are the current shape.

## Shape

```markdown
## Day N — [MM/DD/YYYY]

**Done:**
- <what was actually built or changed, one line each>

**<learning section>**
- <the concept, in your own words>
```

The heading, the day number, and the section names come from the file, not from this skill:

- **Day number** — continue from the last entry. Entries sometimes cover two days (`## Day 12 y 13`);
  match that if the work spanned more than one.
- **Learning section** — early entries use `**Learned:**`, later ones use
  `**Key concepts I can now explain:**` or `**Architecture decisions**`. Pick whichever fits what the
  session actually produced; prefer the most recent entry's choice when it's a toss-up.
- Some entries carry `**Things I've seen but don't deeply understand yet (and that's OK):**` and
  `**Tomorrow:**`. Include them only if there's real content — never pad them.

## Writing it

Draw only from **this session's actual work**. Read the diff (`git diff`, `git log` since the last
entry) rather than reconstructing from memory.

Write in the author's voice, which the existing entries establish clearly: first person, plain,
willing to say what isn't understood yet. Architecture notes state the alternative that was rejected
and why — `create_agent over the deprecated langgraph.prebuilt.create_react_agent to ride the
official direction` is the register to match, not a changelog line.

Do not invent learnings. If the session was mechanical, `**Done:**` alone is a complete entry.

Show the proposed entry and wait for approval before appending.
