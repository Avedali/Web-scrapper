"""
ask_agent.py -- Mode 4: Teammate Q&A interface.
================================================
Run: python agent.py ask --key KEY "what did we find about junior developers?"
Or:  launcher.py → option 8

This is the mode your teammates will use most. They don't need to
understand Python, statistics, or NLP. They just ask a question in
plain English and get a plain-English answer based on the research.

TWO MODES:

  Single question (from the launcher or command line):
    "what did we find about junior developers?"
    → one answer, then done

  Interactive mode (type multiple questions in a row):
    > what's our thesis?
    > summarize the bias check findings
    > give me 3 bullet points for slide 2
    > explain the chi-squared result
    > quit

HOW IT WORKS:
  1. Load ALL scratchpad files (research, analysis, stats, synthesis)
     IN FULL, plus the raw source texts from corpus.json
  2. If it all fits in one AI call, send it as "context" directly.
     Otherwise BATCH it: split the research into batches, ask the AI to
     pull out what's relevant to the question from each batch (several
     batches at once), then answer from those combined notes.
  3. The AI answers the question using ONLY that context
  4. It doesn't make stuff up -- if the research doesn't cover something,
     it says so

Batching means nothing gets cut off: the old version truncated the
research to the first 8000 characters, so the bias audit, red team and
raw sources never reached the AI. The cost is more AI calls per question
(one per batch, plus one to answer).

This is like having a research assistant who read all your notes
and can summarize any part of them on demand.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import os    # File/folder operations and environment variables
import json  # For loading JSON analysis results

from pathlib import Path  # Nice file path handling

from core import (
    call_ai,       # AI calls
    make_batches,  # Pack the research into prompt-sized batches
    ai_map,        # Send one prompt per batch, several at once
)


# ════════════════════════════════════════════════════════════════════════
# BATCHING SETTINGS -- tune these if answers are slow or hit rate limits
# ════════════════════════════════════════════════════════════════════════

BATCH_CHARS = 12000      # Characters of research per AI call (~3000 tokens)
NOTES_CHARS = 24000      # Max characters of combined notes for the final answer
MAX_WORKERS = 4          # Batches read at the same time (lower if rate-limited)
INCLUDE_SOURCES = True   # Also search the raw source texts in corpus.json


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
    # This is the key step: we read EVERYTHING from the scratchpad
    # and feed it to the AI as background knowledge. The AI can only
    # answer from this context -- it can't use its general training
    # data, which prevents hallucination.
    #
    # args.corpus is the path to the scratchpad folder (default: "scratchpad")
    parts = _load_context(args.corpus)

    if not parts:
        print("No research found. Run 'python agent.py explore' first.")
        return

    # Pack the research into batches of up to BATCH_CHARS characters
    batches = make_batches(parts, BATCH_CHARS)

    total = sum(len(b) for b in batches)
    print(f"   Loaded {total} chars of context from scratchpad "
          f"({len(batches)} batch{'es' if len(batches) != 1 else ''}).")

    # ── Build the system prompt ─────────────────────────────────────
    # The system prompt tells the AI WHO it is and HOW to behave.
    # It's like giving instructions to a human assistant:
    # "You're helping business students, speak plainly, cite the research."
    #
    # The research itself is added per question by _answer() -- either
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
        # We send it to the AI, print the answer, and we're done.
        #
        # temp=0.4 is slightly creative but mostly factual.
        # Lower temperature = more predictable/factual answers.
        answer = _answer(args.question, batches, rules, api_key, args.model)
        print(f"\n{answer}")

    else:
        # ── Interactive mode (with memory) ──────────────────────────
        # No question was given upfront → start a conversation loop.
        # The user can ask multiple questions in a row:
        #   > what's our thesis?
        #   > elaborate on that
        #   > give me bullet points for the workforce slide
        #   > quit
        #
        # MEMORY: We keep the last 5 question-answer pairs and include
        # them with each new question. This way "elaborate on that"
        # works -- the AI remembers what "that" refers to.
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
            answer = _answer(full_prompt, batches, rules, api_key, args.model)

            # Print the answer
            print(f"\n{answer}\n")

            # Add this Q&A pair to the history.
            # We truncate long answers to 500 chars in the history to
            # save space (the full answer was already printed).
            conversation_history.append(
                f"Q: {q}\nA: {answer[:500]}"
            )


# ════════════════════════════════════════════════════════════════════════
# ANSWERING -- one call if the research fits, map-then-answer if not
# ════════════════════════════════════════════════════════════════════════

def _answer(prompt, batches, rules, api_key, model):
    """
    Answer one question (prompt may include the conversation history).

    One batch → send it all as context, exactly like a normal question.
    Several batches → two steps:
      1. EXTRACT: for each batch, ask the AI to copy out only what's
         relevant to the question (or say NONE). Batches run in parallel.
      2. ANSWER: send the combined notes as the context and answer.
    """
    if len(batches) == 1:
        return call_ai(prompt, api_key, model, temp=0.4,
                       system=rules + "RESEARCH CONTEXT:\n" + batches[0])

    print(f"   Reading {len(batches)} batches of research...")

    # ai_map keeps the results in batch order (priority files first)
    notes = ai_map(
        batches,
        lambda batch: (
            "Below is one part of a research team's notes and sources.\n"
            "Copy out every finding, number, quote, or source detail that "
            "helps answer the question. Keep the file labels and [n] source "
            "tags. Be brief but don't drop specifics. If nothing in this "
            "part is relevant, reply with exactly: NONE\n\n"
            f"QUESTION:\n{prompt}\n\n"
            f"RESEARCH (one part):\n{batch}"),
        api_key, model, workers=MAX_WORKERS,
    )

    # call_ai returns "" when a call fails -- warn, but answer with the rest
    failed = sum(1 for n in notes if not n)
    if failed:
        print(f"   {failed}/{len(batches)} batches failed -- answer may be incomplete")

    relevant = [n.strip() for n in notes
                if n and n.strip().strip(".").upper() != "NONE"]
    context = "\n\n".join(relevant)[:NOTES_CHARS] or "(Nothing relevant was found.)"

    return call_ai(prompt, api_key, model, temp=0.4,
                   system=rules + "RELEVANT NOTES FROM THE RESEARCH:\n" + context)


# ════════════════════════════════════════════════════════════════════════
# CONTEXT LOADER -- reads all research into labelled parts
# ════════════════════════════════════════════════════════════════════════

def _load_context(corpus_dir):
    """
    Load all research as a list of (label, text) parts for the AI.

    Files are loaded in PRIORITY ORDER -- the most important files
    first. When everything fits in one batch, the AI pays more attention
    to text at the beginning; when batching, the first batches hold the
    core findings.

    Priority order:
      1. Synthesis (the thesis and written sections -- the core findings)
      2. Analysis report (sentiment, topics, NER results)
      3. Stats report (statistical test results)
      4. Simulation report
      5. Bias audit (what's skewed in the research)
      6. Red team critique (what's wrong with the research)
      7. Definitions (key terms)
      8. Decomposition (research questions)
      9. Everything else (remaining scratchpad files)
     10. Analysis results JSON (raw numbers)
     11. Raw source texts from corpus.json (if INCLUDE_SOURCES)

    Nothing is truncated here -- batching handles the size.
    Returns [] if the scratchpad is empty.
    """

    parts = []  # Will hold (label, text) pairs

    pad_dir = Path(corpus_dir)  # Convert string to Path object

    # If the scratchpad folder doesn't exist, nothing to load
    if not pad_dir.exists():
        return []

    # ── Load priority files first ───────────────────────────────────
    priority = [
        "synthesis_01_theses",     # The proposed theses
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

    # ── Load the raw sources (what the findings are based on) ───────
    # With batching these fit too, so the agent can answer "what did
    # source [12] actually say?" -- not just what the summaries say.
    corpus_path = pad_dir / "corpus.json"
    if INCLUDE_SOURCES and corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        for item in corpus:
            parts.append((
                f"source [{item.get('idx', '?')}][{item.get('lang', '?')}]",
                f"{item.get('title', '')} ({item.get('domain', '')}) {item.get('url', '')}\n"
                f"{item.get('content', '')}",
            ))

    return parts
