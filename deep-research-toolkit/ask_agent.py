"""
ask_agent.py: Mode 4, teammate Q&A interface.
================================================
Run: python agent.py ask --key KEY "what did we find about junior developers?"
Or:  launcher.py → option 8

This is the mode your teammates will use most. They don't need to
understand Python, statistics, or NLP. They type a question in plain
English and get a plain-English answer based on the research.

TWO MODES:

  Single question (from the launcher or command line):
    "what did we find about junior developers?"
    → one answer, then done

  Interactive mode (type multiple questions in a row):
    > what's our main finding?
    > summarize the bias check findings
    > give me 3 bullet points for slide 2
    > explain the chi-squared result
    > quit

HOW IT WORKS:
  1. Load all scratchpad files (research, analysis, stats, synthesis)
     in full, plus the raw source texts from corpus.json
  2. If it all fits in one AI call, send it as "context" directly.
     Otherwise batch it: split the research into batches, ask the AI to
     pull out what's relevant to the question from each batch (several
     batches at once), then answer from those combined notes.
  3. The AI answers the question using only that context
  4. If the research doesn't cover something, the AI says so instead of
     making up an answer

Batching means nothing gets cut off: the old version truncated the
research to the first 8000 characters, so the bias audit, red team and
raw sources never reached the AI. The cost is more AI calls per question
(one per batch, plus one to answer).

As a result, the mode can summarize any part of the team's notes
and sources on request.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import os    # File/folder operations and environment variables
import json  # For loading JSON analysis results
import re    # Finding [n] source tags in the notes

from collections import Counter  # Count which sources the notes mention most

from pathlib import Path  # File paths as objects (pad_dir / "corpus.json")

from core import (
    call_ai,       # AI calls
    make_batches,  # Pack the research into prompt-sized batches
    ai_map,        # Send one prompt per batch, several at once
    load_evidence_cards,  # One-line English summary per source (if built)
    NO_EVIDENCE,          # Card text for sources with nothing usable
)


# ════════════════════════════════════════════════════════════════════════
# BATCHING SETTINGS: tune these if answers are slow or hit rate limits
# ════════════════════════════════════════════════════════════════════════

BATCH_CHARS = 40000        # Characters of research per AI call (~10k tokens)
NOTES_CHARS = 200000       # Max characters of combined notes for the final answer
MAX_WORKERS = 4            # Batches read at the same time (lower if rate-limited)
FULL_TEXT_SOURCES = 15     # Sources read in full for each question (stage 2)
INCLUDE_SOURCES = True     # If no evidence cards exist yet, read every raw source


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_ask(args):
    """
    Answer questions about the research in plain language.
    Works in single-question mode or interactive (multi-question) mode.
    """

    # ── Setup ───────────────────────────────────────────────────────
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key found -- run install.py to add one (or pass --key KEY)")
        return

    print("=" * 60)
    print("MODE 4: ASK THE AGENT")
    print("=" * 60)

    # ── Load all research as context ────────────────────────────────
    # This step reads every file in the scratchpad and gives it to the
    # AI as background knowledge. The AI answers only from this context
    # and not from its general training data, which prevents hallucination.
    #
    # args.corpus is the path to the scratchpad folder (default: "scratchpad")
    parts, corpus = _load_context(args.corpus)

    if not parts:
        print("No research found. Run 'python agent.py explore' first.")
        return

    # Pack the research into batches of up to BATCH_CHARS characters
    batches = make_batches(parts, BATCH_CHARS)

    total = sum(len(b) for b in batches)
    print(f"   Loaded {total} chars of context from scratchpad "
          f"({len(batches)} batch{'es' if len(batches) != 1 else ''}).")

    # ── Build the system prompt ─────────────────────────────────────
    # The system prompt tells the AI what role it has and how to behave,
    # for example: "You're helping business students, speak plainly,
    # cite the research."
    #
    # The research itself is added per question by _answer(): either
    # the whole thing (one batch) or the notes pulled from every batch.
    rules = (
        "You are a research assistant helping a team of business students "
        "understand their capstone research. The team researched how AI coding "
        "assistants are transforming software development.\n\n"

        "RULES:\n"
        "- Answer in plain language. No jargon without explanation.\n"
        "- Be specific -- cite findings from the research, not general knowledge.\n"
        "- If asked for slide content, give concise bullet points.\n"
        "- If asked to explain a statistical test, explain what it tested, "
        "what the result was, and what it means in plain English.\n"
        "- If the research doesn't cover something, say so.\n\n"
    )

    # ── Decide which mode to use ────────────────────────────────────

    if args.question:
        # ── Single question mode ────────────────────────────────────
        # The user passed a question as an argument:
        #   python agent.py ask --key KEY "what did we find?"
        # Or typed one in the launcher.
        #
        # The question goes to the AI, the answer is printed, and the mode ends.
        #
        # temp=0.4 is slightly creative but mostly factual.
        # Lower temperature = more predictable/factual answers.
        answer = _answer(args.question, batches, corpus, rules, api_key, args.model)
        print(f"\n{answer}")

    else:
        # ── Interactive mode (with memory) ──────────────────────────
        # No question was given upfront → start a conversation loop.
        # The user can ask multiple questions in a row:
        #   > what's our main finding?
        #   > elaborate on that
        #   > give me bullet points for the workforce slide
        #   > quit
        #
        # MEMORY: the last 5 question-answer pairs are sent with each
        # new question. That makes "elaborate on that" work, because the
        # AI can see what "that" refers to.
        #
        # The memory is a list of strings. We cap it at 5 pairs to
        # avoid exceeding the AI's input limit (the research context
        # already takes up most of the space).

        print("\n   Ask anything about the research. Type 'quit' to exit.")
        print("   The agent remembers your last 5 questions.\n")

        # conversation_history holds the last few Q&A pairs as strings.
        # We send these along with each new question so the AI has
        # context about what was discussed before.
        conversation_history = []  # List of "Q: ... A: ..." strings
        MAX_MEMORY = 5             # How many Q&A pairs to remember

        # "while True" = keep looping forever (until we "break" out)
        while True:
            try:
                # Show a prompt ("> ") and wait for the user to type
                q = input("   > ").strip()

            except (EOFError, KeyboardInterrupt):
                break

            # Check if the user wants to quit
            if not q or q.lower() in ("quit", "exit", "q"):
                break

            # Build the prompt with conversation history.
            # The history gives the AI context about previous questions
            # so "elaborate on that" or "say more about point 2" works.
            if conversation_history:
                # Join the last N pairs into one text block
                history_text = "\n".join(conversation_history[-MAX_MEMORY:])
                full_prompt = (
                    f"PREVIOUS CONVERSATION:\n{history_text}\n\n"
                    f"NEW QUESTION: {q}"
                )
            else:
                full_prompt = q

            # Send to the AI
            answer = _answer(full_prompt, batches, corpus, rules, api_key, args.model)

            # Print the answer
            print(f"\n{answer}\n")

            # Add this Q&A pair to the history.
            # We truncate long answers to 500 chars in the history to
            # save space (the full answer was already printed).
            conversation_history.append(
                f"Q: {q}\nA: {answer[:500]}"
            )


# ════════════════════════════════════════════════════════════════════════
# ANSWERING: one call if the research fits, map-then-answer if not
# ════════════════════════════════════════════════════════════════════════

def _extract_prompt(question):
    """Prompt that pulls the notes relevant to `question` out of one batch."""
    return lambda batch: (
        "Below is one part of a research team's notes and sources.\n"
        "Copy out every finding, number, quote, or source detail that helps "
        "answer the question. Keep the file labels and [n] source tags (they "
        "tell us which sources to read in full). Be brief but don't drop "
        "specifics. If nothing in this part is relevant, reply with exactly: NONE\n\n"
        f"QUESTION:\n{question}\n\nRESEARCH (one part):\n{batch}")


def _answer(prompt, batches, corpus, rules, api_key, model):
    """
    Answer one question (prompt may include the conversation history).

    Stage 1: read the findings + one evidence card per source (in batches,
             several at once) and copy out what's relevant to the question.
    Stage 2: read the full text of the sources those notes point to most
             (up to FULL_TEXT_SOURCES), so answers can use details that
             only appear deep inside a source.
    Then answer from all the notes. Nothing is truncated: long material is
    split into batches rather than cut off.
    """
    print(f"   Reading {len(batches)} batches of findings and evidence cards...")
    notes = ai_map(batches, _extract_prompt(prompt), api_key, model, workers=MAX_WORKERS)

    # Stage 2: count how often the notes cite each source with an [n] tag
    by_idx = {s.get("idx"): s for s in corpus}
    mentions = Counter(int(n) for note in notes if note
                       for n in re.findall(r"\[(\d+)\]", note) if int(n) in by_idx)
    picks = [i for i, _ in mentions.most_common(FULL_TEXT_SOURCES)]
    if picks:
        print(f"   Reading the full text of {len(picks)} relevant sources...")
        parts = [(f"source [{i}][{by_idx[i].get('lang', '?')}] {by_idx[i].get('title', '')} "
                  f"({by_idx[i].get('domain', '')}) {by_idx[i].get('url', '')}",
                  by_idx[i].get("content", "")) for i in picks]
        notes += ai_map(make_batches(parts, BATCH_CHARS), _extract_prompt(prompt),
                        api_key, model, workers=MAX_WORKERS)

    # call_ai returns "" when a call fails: warn, then answer with the rest
    failed = sum(1 for n in notes if not n)
    if failed:
        print(f"   {failed}/{len(notes)} reading calls failed -- answer may be incomplete")

    relevant = [n.strip() for n in notes
                if n and n.strip().strip(".").upper() != "NONE"]
    context = "\n\n".join(relevant)[:NOTES_CHARS] or "(Nothing relevant was found.)"

    return call_ai(prompt, api_key, model, temp=0.4,
                   system=rules + "RELEVANT NOTES FROM THE RESEARCH:\n" + context)


# ════════════════════════════════════════════════════════════════════════
# CONTEXT LOADER: reads all research into labelled parts
# ════════════════════════════════════════════════════════════════════════

def _load_context(corpus_dir):
    """
    Load all research as a list of (label, text) parts for the AI.

    Files are loaded in priority order, with the most important files
    first. When everything fits in one batch, the AI pays more attention
    to text at the beginning; when batching, the first batches hold the
    core findings.

    Priority order:
      1. Synthesis (the research question, themes and written sections)
      2. Analysis report (sentiment, topics, NER results)
      3. Stats report (statistical test results)
      4. Simulation report
      5. Bias audit (what's skewed in the research)
      6. Red team critique (what's wrong with the research)
      7. Definitions (key terms)
      8. Decomposition (research questions)
      9. Everything else (remaining scratchpad files)
     10. Analysis results JSON (raw numbers)
     11. One evidence card per source or, if no cards were built yet,
         the raw source texts (if INCLUDE_SOURCES)

    Nothing is truncated here; batching handles the size.
    Returns (parts, corpus): parts is [] if the scratchpad is empty;
    corpus is the source list (for reading full texts in stage 2).
    """

    parts = []  # Will hold (label, text) pairs

    pad_dir = Path(corpus_dir)  # Convert string to Path object

    # If the scratchpad folder doesn't exist, nothing to load
    if not pad_dir.exists():
        return [], []

    # ── Load priority files first ───────────────────────────────────
    priority = [
        "synthesis_01_framing",    # The research question and themes
        "synthesis_03_brief_raw",  # Raw brief sections
        "analyze_report",          # NLP analysis summary
        "stats_report",            # Statistical test results
        "simulation_report",       # Simulation findings
        "065_bias",                # Bias audit
        "06_red_team",             # Red team critique
        "01_definitions",          # Key term definitions
        "02_decomposition",        # Research questions by language
    ]

    # Track which files we've already loaded (to avoid duplicates later)
    loaded = set()

    for name in priority:
        p = pad_dir / f"{name}.md"
        if p.exists():
            # Label each part with its filename so the AI can say
            # "according to the bias audit, ..."
            parts.append((name, p.read_text(encoding="utf-8")))
            loaded.add(p.name)

    # ── Load remaining scratchpad files ─────────────────────────────
    # deepen reports, gap analyses, etc. sorted() = alphabetical, which
    # (because of the numbered prefixes) is also chronological order.
    loaded.add("synthesis_00_evidence_cards.md")  # Cards are added below
    for p in sorted(pad_dir.glob("*.md")):
        if p.name not in loaded:
            parts.append((p.stem, p.read_text(encoding="utf-8")))

    # ── Load analysis results JSON (raw numbers) ────────────────────
    # The AI can read numbers from it to give specific answers like
    # "the average sentiment of vendor sources was +0.34."
    results_path = pad_dir / "analysis_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        parts.append(("analysis_results", json.dumps(data, indent=2)))

    # ── The sources themselves ──────────────────────────────────────
    corpus = []
    corpus_path = pad_dir / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

    cards = load_evidence_cards(corpus_dir)
    if cards:
        # One line per source. The full texts are read in stage 2, only
        # for the sources relevant to the question (see _answer)
        by_idx = {s.get("idx"): s for s in corpus}
        parts.append(("evidence cards (one per source)", "\n".join(
            f"[{i}][{by_idx.get(i, {}).get('lang', '?')}] {c}"
            for i, c in sorted(cards.items()) if not c.startswith(NO_EVIDENCE))))
    elif INCLUDE_SOURCES:
        # No cards yet (analyze/synthesize haven't run): read every raw
        # source instead. This is slower and costs more for big corpora.
        for item in corpus:
            parts.append((
                f"source [{item.get('idx', '?')}][{item.get('lang', '?')}]",
                f"{item.get('title', '')} ({item.get('domain', '')}) {item.get('url', '')}\n"
                f"{item.get('content', '')}",
            ))

    return parts, corpus
