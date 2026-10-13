"""ask_agent.py: Mode 4, plain-English Q&A over the team's research.
Run: python agent.py ask --key KEY "question" (no question starts interactive mode).
How it works: notes relevant to the question are pulled from every batch and the most-cited sources, then the AI answers from those notes only."""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import os
import json
import re

from collections import Counter

from pathlib import Path

from core import (
    call_ai,
    make_batches,
    ai_map,
    load_evidence_cards,
    NO_EVIDENCE,
)


# ════════════════════════════════════════════════════════════════════════
# BATCHING SETTINGS
# ════════════════════════════════════════════════════════════════════════

BATCH_CHARS = 40000        # Characters of research per AI call
NOTES_CHARS = 200000       # Max characters of notes for the final answer
MAX_WORKERS = 4            # Batches read in parallel
FULL_TEXT_SOURCES = 15     # Sources read in full per question
INCLUDE_SOURCES = True     # Read raw sources when no evidence cards exist


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_ask(args):
    """Answer one question, or loop over questions in interactive mode."""

    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key found -- run install.py to add one (or pass --key KEY)")
        return

    print("=" * 60)
    print("MODE 4: ASK THE AGENT")
    print("=" * 60)

    parts, corpus = _load_context(args.corpus)

    if not parts:
        print("No research found. Run 'python agent.py explore' first.")
        return

    batches = make_batches(parts, BATCH_CHARS)

    total = sum(len(b) for b in batches)
    print(f"   Loaded {total} chars of context from scratchpad "
          f"({len(batches)} batch{'es' if len(batches) != 1 else ''}).")

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

    if args.question:
        answer = _answer(args.question, batches, corpus, rules, api_key, args.model)
        print(f"\n{answer}")

    else:
        print("\n   Ask anything about the research. Type 'quit' to exit.")
        print("   The agent remembers your last 5 questions.\n")

        conversation_history = []
        MAX_MEMORY = 5

        while True:
            try:
                q = input("   > ").strip()

            except (EOFError, KeyboardInterrupt):
                break

            if not q or q.lower() in ("quit", "exit", "q"):
                break

            if conversation_history:
                history_text = "\n".join(conversation_history[-MAX_MEMORY:])
                full_prompt = (
                    f"PREVIOUS CONVERSATION:\n{history_text}\n\n"
                    f"NEW QUESTION: {q}"
                )
            else:
                full_prompt = q

            answer = _answer(full_prompt, batches, corpus, rules, api_key, args.model)

            print(f"\n{answer}\n")

            conversation_history.append(
                f"Q: {q}\nA: {answer[:500]}"
            )


# ════════════════════════════════════════════════════════════════════════
# ANSWERING
# ════════════════════════════════════════════════════════════════════════

def _extract_prompt(question):
    """Return a prompt builder that pulls notes relevant to `question` from one batch."""
    return lambda batch: (
        "Below is one part of a research team's notes and sources.\n"
        "Copy out every finding, number, quote, or source detail that helps "
        "answer the question. Keep the file labels and [n] source tags (they "
        "tell us which sources to read in full). Be brief but don't drop "
        "specifics. If nothing in this part is relevant, reply with exactly: NONE\n\n"
        f"QUESTION:\n{question}\n\nRESEARCH (one part):\n{batch}")


def _answer(prompt, batches, corpus, rules, api_key, model):
    """Extract notes from all batches, then from the most-cited full sources, and answer from them."""
    print(f"   Reading {len(batches)} batches of findings and evidence cards...")
    notes = ai_map(batches, _extract_prompt(prompt), api_key, model, workers=MAX_WORKERS)

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

    failed = sum(1 for n in notes if not n)
    if failed:
        print(f"   {failed}/{len(notes)} reading calls failed -- answer may be incomplete")

    relevant = [n.strip() for n in notes
                if n and n.strip().strip(".").upper() != "NONE"]
    context = "\n\n".join(relevant)[:NOTES_CHARS] or "(Nothing relevant was found.)"

    return call_ai(prompt, api_key, model, temp=0.4,
                   system=rules + "RELEVANT NOTES FROM THE RESEARCH:\n" + context)


# ════════════════════════════════════════════════════════════════════════
# CONTEXT LOADER
# ════════════════════════════════════════════════════════════════════════

def _load_context(corpus_dir):
    """Return (parts, corpus): (label, text) parts in priority order, ending with evidence cards or raw sources."""

    parts = []

    pad_dir = Path(corpus_dir)

    if not pad_dir.exists():
        return [], []

    priority = [
        "synthesis_01_framing",
        "synthesis_03_brief_raw",
        "analyze_report",
        "stats_report",
        "simulation_report",
        "065_bias",
        "06_red_team",
        "01_definitions",
        "02_decomposition",
    ]

    loaded = set()

    for name in priority:
        p = pad_dir / f"{name}.md"
        if p.exists():
            parts.append((name, p.read_text(encoding="utf-8")))
            loaded.add(p.name)

    loaded.add("synthesis_00_evidence_cards.md")  # Cards are added below
    for p in sorted(pad_dir.glob("*.md")):
        if p.name not in loaded:
            parts.append((p.stem, p.read_text(encoding="utf-8")))

    results_path = pad_dir / "analysis_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        parts.append(("analysis_results", json.dumps(data, indent=2)))

    corpus = []
    corpus_path = pad_dir / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

    cards = load_evidence_cards(corpus_dir)
    if cards:
        by_idx = {s.get("idx"): s for s in corpus}
        parts.append(("evidence cards (one per source)", "\n".join(
            f"[{i}][{by_idx.get(i, {}).get('lang', '?')}] {c}"
            for i, c in sorted(cards.items()) if not c.startswith(NO_EVIDENCE))))
    elif INCLUDE_SOURCES:
        for item in corpus:
            parts.append((
                f"source [{item.get('idx', '?')}][{item.get('lang', '?')}]",
                f"{item.get('title', '')} ({item.get('domain', '')}) {item.get('url', '')}\n"
                f"{item.get('content', '')}",
            ))

    return parts, corpus
