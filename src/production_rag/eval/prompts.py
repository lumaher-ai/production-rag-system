"""The generator prompts, and the schemas that constrain their replies.

``PROMPT_VERSION`` is part of every generation unit's id, so bumping it makes a
re-run regenerate everything rather than resume — which is the behaviour you
want, because questions written under two different prompts are not the same
population and averaging them together hides whichever one is worse.

Each prompt states that its rules are checked programmatically. That is not
flattery of the model: compliance measurably improves when a constraint is
described as verified, and every rule stated below genuinely is verified in
``gates.py``. The prompt is a hint; the gate is the guarantee.
"""

PROMPT_VERSION = "eval-gen-v1"

# ─── Paraphrase + exact-term, one chunk at a time ───

SINGLE_CHUNK_SYSTEM = """\
You write evaluation questions for a document retrieval system. You are given ONE
passage from a real document. You produce questions a user would type into a
search box, together with the exact sentence(s) from the passage that answer them.

Every rule below is checked programmatically. A violation discards the item.

1. "snippet" MUST be copied character-for-character from the passage. Do not fix
   typos, do not re-wrap lines, do not expand abbreviations, do not add ellipses,
   do not summarise. One contiguous span, 20-400 characters, that actually
   contains the answer. Never include the <<<PASSAGE>>> delimiter lines.
2. The question must stand on its own. The reader has never seen this passage and
   does not know it exists. Never write "according to the document", "in the
   passage", "in the text above", "this section", "as mentioned", or anything
   equivalent. Never use a pronoun whose antecedent appears only in the passage
   ("what does it configure?").
3. The question must not contain its own answer. Never quote eight or more
   consecutive words from the snippet.
4. "answer" is a direct, self-contained answer of one or two sentences, fully
   derivable from "snippet" alone.
5. Ask about substance. Never ask about the passage's formatting, its length, its
   position, or how many bullet points it has.

Produce a mix of exactly two types:

- "paraphrase" — ask about a fact in the passage using DIFFERENT vocabulary from
  the passage. If the passage says "iterative scan re-scans until the limit is
  satisfied", a paraphrase question asks "how does the vector index avoid
  returning fewer results than requested when filters narrow the candidates?"
  The words differ; the meaning does not. This type tests semantic retrieval, so
  lexical overlap with the passage is a defect, not a convenience.
- "exact_term" — ask a question whose natural phrasing contains a literal
  identifier copied from the passage: a setting name, a version string, a
  function or column name, an error code, a number, a proper noun, an acronym.
  Spell that token exactly as the passage spells it and repeat it in the
  "exact_term" field. This type tests lexical retrieval. If the passage contains
  no such identifier, produce no exact_term items for it. Do not invent one.

Produce 3 to 5 items in total, aiming for roughly half of each type. If the
passage is boilerplate, a table of contents, a list of links, a heading with no
body, or otherwise carries no answerable fact, return {"items": []}. An empty
list is a correct and expected answer. Padding it is not.

Return ONLY JSON. No prose, no code fence, no commentary.\
"""

SINGLE_CHUNK_USER = """\
Document title: {document_title}
Section: {section}

<<<PASSAGE>>>
{content}
<<<END PASSAGE>>>\
"""

SINGLE_CHUNK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "eval_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query_type", "question", "answer", "snippet", "exact_term"],
                        "properties": {
                            "query_type": {"type": "string", "enum": ["paraphrase", "exact_term"]},
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "snippet": {"type": "string"},
                            "exact_term": {"type": ["string", "null"]},
                        },
                    },
                }
            },
        },
    },
}

# ─── Multi-hop, two chunks at a time ───

MULTI_HOP_SYSTEM = """\
You write MULTI-HOP evaluation questions for a document retrieval system. You are
given TWO passages, A and B. Write a question that CANNOT be answered by either
passage alone — answering it requires a fact from A and a fact from B.

Apply this test to every question you write: if a reader were handed only passage
A, could they answer it? If yes, the question is wrong. Now the same for B. If no
genuine two-passage question exists here, return {"items": []}. That is the right
answer far more often than not. An invented multi-hop question is worse than no
question, because it is scored as a retrieval failure when the system is behaving
correctly.

Rules, all checked programmatically:

1. "snippet_a" is copied character-for-character from passage A and "snippet_b"
   from passage B. Each is one contiguous span of 20-400 characters, unedited,
   excluding the delimiter lines.
2. Both snippets must be load-bearing. If removing one still leaves the question
   answerable, the item is invalid.
3. The question stands on its own — no "according to the passages", no "passage
   A", no pronouns pointing at either passage. The reader has seen neither.
4. "answer" is one to three sentences and combines both facts.
5. "why_both_needed" states in one sentence which fact comes from A and which
   comes from B. Be specific. "Both provide context" is a rejection.

Shapes that work: comparison ("which of X and Y is larger / stricter / earlier"),
composition (A defines a term, B gives a value for it), causal chain (A states a
cause, B states its consequence), constraint check (A gives a limit, B gives a
quantity that must fit inside it).

Produce 1 to 2 items. Return ONLY JSON. No prose, no code fence.\
"""

MULTI_HOP_USER = """\
Document title: {document_title}

<<<PASSAGE A>>>
{content_a}
<<<END PASSAGE A>>>

<<<PASSAGE B>>>
{content_b}
<<<END PASSAGE B>>>\
"""

MULTI_HOP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "multi_hop_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "question",
                            "answer",
                            "snippet_a",
                            "snippet_b",
                            "why_both_needed",
                        ],
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "snippet_a": {"type": "string"},
                            "snippet_b": {"type": "string"},
                            "why_both_needed": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}

# ─── Unanswerable, seeded from a real chunk ───

UNANSWERABLE_SYSTEM = """\
You write NEGATIVE evaluation questions for a document retrieval system. You are
given one passage. Write questions that sound like they belong to this material —
same topic, same entities, same vocabulary — but whose answer is NOT in it and,
to the best of your judgement, is not anywhere in a document that contains it.

These questions test whether a system correctly says "I don't know". A question
the corpus CAN answer is worse than useless here: it converts correct behaviour
into a scored failure. When in doubt, do not produce the item.

Shapes that reliably work:
- ask for a specific quantity the passage discusses only qualitatively — a
  latency, a throughput, a cost, a percentage it never states;
- ask about a named entity in the passage along a dimension the material never
  covers: its price, its author, its release date, its licence, its vendor;
- ask for a comparison against a system, product, or version never mentioned;
- ask "why was X rejected" or "when was X removed" about something the passage
  never rejects or removes.

Never invent an entity that does not appear in the passage — an entirely
off-topic question belongs to a different set and will be discarded. Every
question must reuse at least one term from the passage, spelled as it is spelled
there.

Also: no "according to the document"; no pronouns pointing at the passage; 15-300
characters; "why_absent" states in one sentence why this passage cannot answer it.

Produce 1 to 3 items, or {"items": []}. Return ONLY JSON. No prose, no fence.\
"""

UNANSWERABLE_USER = SINGLE_CHUNK_USER

UNANSWERABLE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "unanswerable_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["question", "why_absent"],
                        "properties": {
                            "question": {"type": "string"},
                            "why_absent": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}

# ─── The verifier: a different model, deliberately ───

VERIFIER_SYSTEM = """\
You decide exactly one thing: whether the passages below contain the information
needed to answer the question. Not whether you know the answer — whether THESE
passages state it. Partial information that does not settle the question counts
as NO.

Reply with a single word: YES or NO.\
"""

VERIFIER_USER = """\
QUESTION: {question}

PASSAGES:
{passages}\
"""

REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON. Return only the JSON object, "
    "with no prose, no explanation, and no code fence."
)
