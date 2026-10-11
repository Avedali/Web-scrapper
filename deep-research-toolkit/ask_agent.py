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
  2. Feed them to the AI as "context" (background knowledge)
  3. The AI answers the question using ONLY that context
  4. It doesn't make stuff up -- if the research doesn't cover something,
     it says so

This is like having a research assistant who read all your notes
and can summarize any part of them on demand.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import os    # File/folder operations and environment variables
import json  # For loading JSON analysis results

from pathlib import Path  # Nice file path handling

from core import call_ai  # AI calls


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
        print("--key required")
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
    context = _load_context(args.corpus)

    if not context:
        print("No research found. Run 'python agent.py explore' first.")
        return

    # len(context) = number of characters. A typical research corpus
    # produces 10,000-30,000 characters of context.
    print(f"   Loaded {len(context)} chars of context from scratchpad.")

    # ── Build the system prompt ─────────────────────────────────────
    # The system prompt tells the AI WHO it is and HOW to behave.
    # It's like giving instructions to a human assistant:
    # "You're helping business students, speak plainly, cite the research."
    #
    # CRITICAL: We append the entire research context to the system
    # prompt. This means every question the user asks gets answered
    # with the full research as background knowledge.
    #
    # context[:8000] = first 8000 characters of context. We truncate
    # because AI models have a maximum input size. 8000 characters
    # is roughly 2000 words -- enough to cover the key findings.
    system = (
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

        # Here's where the research context is injected.
        # Everything after this line is the actual research data
        # that the AI will reference when answering.
        "RESEARCH CONTEXT:\n" + context[:8000]
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
        answer = call_ai(args.question, api_key, args.model, temp=0.4, system=system)
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
            answer = call_ai(full_prompt, api_key, args.model, temp=0.4, system=system)

            # Print the answer
            print(f"\n{answer}\n")

            # Add this Q&A pair to the history.
            # We truncate long answers to 500 chars in the history to
            # save space (the full answer was already printed).
            conversation_history.append(
                f"Q: {q}\nA: {answer[:500]}"
            )


# ════════════════════════════════════════════════════════════════════════
# CONTEXT LOADER -- reads all scratchpad files into one string
# ════════════════════════════════════════════════════════════════════════

def _load_context(corpus_dir):
    """
    Load all scratchpad files into one big text string for the AI.

    Files are loaded in PRIORITY ORDER -- the most important files
    first, because the AI pays more attention to text at the beginning
    of its context. If we have to truncate (the context is too long
    for the AI's input limit), the most important stuff survives.

    Priority order:
      1. Synthesis (the thesis and written sections -- the core findings)
      2. Analysis report (sentiment, topics, NER results)
      3. Stats report (statistical test results)
      4. Bias audit (what's skewed in the research)
      5. Red team critique (what's wrong with the research)
      6. Definitions (key terms)
      7. Decomposition (research questions)
      8. Everything else (remaining scratchpad files)
      9. Analysis results JSON (raw numbers)

    Returns one big string, or "" if the scratchpad is empty.
    """

    parts = []  # Will hold text chunks from each file

    pad_dir = Path(corpus_dir)  # Convert string to Path object

    # If the scratchpad folder doesn't exist, nothing to load
    if not pad_dir.exists():
        return ""

    # ── Load priority files first ───────────────────────────────────
    # These are the most important files -- they contain the actual
    # findings, not raw data. If space is limited, these survive.
    priority = [
        "synthesis_01_theses", # The proposed theses
        "synthesis_03_brief_raw",  # Raw brief sections
        "analyze_report",     # NLP analysis summary
        "stats_report",       # Statistical test results
        "simulation_report",  # Simulation findings
        "065_bias",           # Bias audit
        "06_red_team",        # Red team critique
        "01_definitions",     # Key term definitions
        "02_decomposition",   # Research questions by language
    ]

    # Track which files we've already loaded (to avoid duplicates later)
    loaded = set()  # A set of filenames we've already read

    for name in priority:
        p = pad_dir / f"{name}.md"  # Build the full path

        if p.exists():
            # Read the file and take the first 2000 characters.
            # We label each chunk with its filename so the AI knows
            # which file each piece of information came from.
            # This helps it give answers like "according to the
            # bias audit, ..."
            parts.append(
                f"── {name} ──\n"                             # Label header
                f"{p.read_text(encoding='utf-8')[:2000]}\n"   # Content (truncated)
            )
            # Mark this file as loaded.
            # We use p.name (just the filename, like "01_definitions.md")
            # rather than the full path, for simpler comparison later.
            loaded.add(p.name)

    # ── Load remaining scratchpad files ─────────────────────────────
    # Any .md file we haven't loaded yet gets a shorter excerpt (1000 chars).
    # These might include deepen reports, gap analyses, etc.
    #
    # sorted() puts them in alphabetical order (which, because of the
    # numbered prefixes, is also chronological order).
    for p in sorted(pad_dir.glob("*.md")):  # Find all .md files
        if p.name not in loaded:             # Skip ones we already loaded
            parts.append(
                f"── {p.stem} ──\n"                             # p.stem = filename without .md
                f"{p.read_text(encoding='utf-8')[:1000]}\n"     # Shorter excerpt
            )

    # ── Load analysis results JSON (raw numbers) ────────────────────
    # This file contains the actual sentiment scores, topic counts, etc.
    # as structured data. The AI can read numbers from it to give
    # specific answers like "the average sentiment of vendor sources
    # was +0.34."
    results_path = pad_dir / "analysis_results.json"

    if results_path.exists():
        with open(results_path) as f:
            # json.load reads the file → Python dictionary
            # json.dumps converts it back to a formatted string (for the AI to read)
            # indent=2 makes it human-readable (and AI-readable)
            # [:2000] truncates to 2000 characters
            data = json.load(f)
            parts.append(
                f"── analysis_results ──\n"
                f"{json.dumps(data, indent=2)[:2000]}\n"
            )

    # Join all parts into one big string, separated by newlines
    return "\n".join(parts)
