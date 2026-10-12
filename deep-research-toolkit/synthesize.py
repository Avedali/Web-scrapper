"""
synthesize.py -- Mode 1.b: Thesis generation + outline + brief.
================================================================
Run: python agent.py synthesize
Or:  launcher.py → option 4

This is where the opinion forms -- BUT only from evidence.
It reads EVERYTHING the previous modes produced (sources, NLP analysis,
statistical tests, red team critiques, bias audits) and:

  1.b.i:   Proposes 3 defensible theses ranked by evidence strength
  1.b.ii:  Generates a paper outline shaped by the chosen thesis
  1.b.iii: Writes each section (grounded in evidence, not memory)
  1.b.iv:  Formats references (APA) and cleans voice (strips LLM-speak)

ONLY run this AFTER explore + analyze + stats.
The thesis comes from the data, not from the researcher's gut.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # Read/write structured data files
import os    # File/folder operations
import re    # Pattern matching (used to parse outlines and clean text)

from datetime import datetime       # Timestamps for the final report
from collections import defaultdict # Auto-initializing dictionary
from pathlib import Path            # Nice file path handling

# From our shared toolbox:
from core import (
    call_ai,               # Send prompts to the AI model
    pad,                   # Save work to the scratchpad
    build_evidence_cards,  # 1-2 sentence English summary per source
    NO_EVIDENCE,           # Card text for sources with nothing usable
    LANGUAGES,             # The 9-language config (for APA language notes)
    BANNED_PHRASES,        # LLM-speak phrases to strip (for voice cleaning)
)


# ════════════════════════════════════════════════════════════════════════
# HOW MUCH EVIDENCE THE AI SEES -- characters per prompt
# ════════════════════════════════════════════════════════════════════════
#
# The old version showed the AI the first 6000 characters of evidence, so a
# 2,500-word brief built from a 121-source corpus cited just 2 sources.
# Now every source is read (as an evidence card), and each section gets the
# full text of the sources assigned to it. Current models read far more than
# this (the default reads ~1M tokens); a full synthesize run costs a few
# cents. Lower these if you pick a model with a small context window.

THESIS_CONTEXT_CHARS = 80000   # Findings + ALL evidence cards for the thesis step
CARDS_CHARS = 60000            # Evidence cards shown when writing each section
SOURCE_CHARS = 2500            # Full text per source assigned to a section
MAX_SOURCES_PER_SECTION = 12   # Sources the evidence map assigns per section
MAX_SECTIONS_PER_SOURCE = 3    # A source is ASSIGNED to at most this many sections
                               # (any section can still cite it from the cards)
EVIDENCE_CHARS = 40000         # Fallback evidence block (if cards fail)
SOURCES_PER_TIER = 15          # Max sources per quality tier in that fallback


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_synthesize(args):
    """
    Run the full synthesis pipeline:
    thesis → outline → write sections → APA references → voice clean → save.
    """

    # ── Setup ───────────────────────────────────────────────────────
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key found -- run install.py to add one (or pass --key KEY)")
        return

    model = args.model  # Which AI model to use

    print("=" * 60)
    print("MODE 1.b: SYNTHESIZE (thesis from evidence)")
    print("=" * 60)

    # ── Load all evidence from the scratchpad ───────────────────────
    # This reads every file that explore, analyze, and stats produced.
    # The AI will use this as its ONLY source of information.
    context = _load_all_evidence(args.corpus)

    if not context:
        print("No evidence found. Run explore → analyze → stats first.")
        return

    # len(context) = number of characters (letters) in the text
    print(f"   Loaded {len(context)} chars of evidence from scratchpad/")

    # ── Read EVERY source: one evidence card each ───────────────────
    # A card is a 1-2 sentence English summary of a source's findings,
    # tagged [n] (see core.build_evidence_cards). ~150 cards fit in one
    # prompt, so the AI can see the whole corpus -- including the
    # Japanese, Korean, Chinese... sources -- not just the first few.
    # Cards are cached: if analyze already built them, this is free.
    corpus = _load_corpus(args.corpus)
    by_idx = {s["idx"]: s for s in corpus}
    cards = build_evidence_cards(corpus, api_key, model, args.corpus) if corpus else {}
    usable = {i: c for i, c in cards.items() if not c.startswith(NO_EVIDENCE)}

    if usable:
        evidence = "\n".join(
            f"[{i}][{by_idx[i].get('lang', '?')}][{by_idx[i].get('quality', '?')}] {c}"
            for i, c in sorted(usable.items()) if i in by_idx)
        print(f"   Read all {len(corpus)} sources -> {len(usable)} evidence cards "
              f"({len(cards) - len(usable)} had no usable evidence)")
    else:
        # No cards (AI calls failed)? Fall back to source snippets
        evidence = _build_evidence_block(args.corpus)[:EVIDENCE_CHARS]
        print("   Evidence cards unavailable -- using source snippets instead")

    pad("synthesis_00_evidence_cards",
        f"# Evidence cards (one per source, cite as [n])\n\n{evidence}")

    # ════════════════════════════════════════════════════════════════
    # 1.b.i: THESIS GENERATION
    # ════════════════════════════════════════════════════════════════
    # This is the critical step. Instead of the researcher deciding
    # what to argue BEFORE looking at data (which creates confirmation
    # bias), we ask the AI to read ALL the evidence and propose
    # the strongest defensible positions.
    #
    # The AI proposes 3 theses, ranked by evidence strength.
    # The user picks which one to use for the paper.
    #
    # A thesis is NOT a topic. "AI is changing software development"
    # is a topic (no one disagrees). "AI coding assistants increase
    # individual productivity but create systemic risks to code quality
    # that organizations are not yet equipped to manage" is a thesis
    # (someone could argue the opposite).

    print(f"\n{'─'*50}\n1.b.i: THESIS GENERATION\n{'─'*50}")
    print("   Reading all evidence and proposing defensible theses...\n")

    # Build the prompt for thesis generation.
    # f"""...""" is a multi-line f-string (formatted string).
    # The prompt gets the findings (context) plus the tagged sources,
    # truncated to THESIS_CONTEXT_CHARS (AI models have a maximum input length).
    thesis_evidence = (context + "\n\nEVIDENCE CARDS (one per source, cite as [n]):\n"
                       + evidence)[:THESIS_CONTEXT_CHARS]
    thesis_prompt = f"""You are a research advisor helping business students write a capstone paper.

You have access to ALL the evidence they collected -- raw sources, sentiment
analysis, statistical tests, topic modeling, bias audits, and red team critiques.

YOUR TASK: Based ONLY on this evidence, propose 3 possible thesis statements
that are:
1. Defensible with the evidence that exists (not aspirational)
2. Arguable (someone could disagree)
3. Specific enough to structure a paper around
4. Relevant to a business audience (not just technical)

For each thesis, explain:
- What evidence supports it (cite specific findings)
- What evidence complicates it (the strongest counterargument)
- How strong the evidence base is (strong / moderate / thin)
- What the paper outline would look like if they chose this thesis

EVIDENCE:
{thesis_evidence}

Do NOT propose generic theses like "AI is transforming software development."
That's a topic, not a thesis. A thesis takes a position: "AI coding assistants
increase individual productivity but create systemic risks to code quality
that organizations are not yet equipped to manage."

Output 3 theses, ranked by strength of evidence."""

    # Send the prompt to the AI
    theses = call_ai(thesis_prompt, api_key, model, temp=0.5)

    # Save to scratchpad so other modes (and teammates via ask) can read it
    pad("synthesis_01_theses", f"# Proposed Theses\n\n{theses}")

    # Show the first 2000 characters to the user so they can review
    print(theses[:2000])

    # ── Ask the user which thesis to use ────────────────────────────
    # This is an interactive step: the script pauses and waits for
    # the user to type 1, 2, or 3 (or just press Enter for #1).
    print(f"\n{'─'*50}")
    print("   Review the theses above. Which one?")
    print("   Press Enter to use thesis #1 (strongest evidence),")
    print("   or type 2 or 3 to pick a different one.")

    try:
        choice = input("   > ").strip()  # Wait for user input
    except (EOFError, KeyboardInterrupt):
        choice = ""  # If they hit Ctrl+C, default to thesis #1

    # Convert their choice to an integer.
    # int(choice) converts "2" → 2, but would crash on "" or "abc".
    # So we only do it if choice is "1", "2", or "3".
    thesis_num = int(choice) if choice in ("1", "2", "3") else 1

    print(f"   → Using thesis #{thesis_num}\n")

    # ════════════════════════════════════════════════════════════════
    # 1.b.ii: OUTLINE GENERATION
    # ════════════════════════════════════════════════════════════════
    # Now that we have a thesis, we generate the paper outline.
    # The outline is shaped BY the thesis -- each section exists
    # because the thesis needs it, not because it's a standard section.
    #
    # Key rule: "No 'Background' section unless background is genuinely
    # necessary for the argument." Standard academic papers often have
    # Background, Literature Review, Methodology, Findings, Discussion,
    # Conclusion -- but that structure is for a topic, not a thesis.
    # A thesis-driven paper has sections that each advance the argument.

    print(f"{'─'*50}\n1.b.ii: OUTLINE FROM THESIS\n{'─'*50}")

    outline_prompt = f"""Based on thesis #{thesis_num} from this analysis, generate a paper outline.

The paper is 8-10 pages, double-spaced, for a business school capstone.
The audience is financial industry professionals, not technologists.

THESES:
{theses[:8000]}

RULES:
- Every section must exist because the thesis NEEDS it, not because
  it's a standard section. No "Background" section unless background
  is genuinely necessary for the argument.
- Each section should have a one-sentence summary of what it argues
  (not what it covers -- what it ARGUES).
- Note which evidence from the scratchpad supports each section.
- Include a "Limitations and counterarguments" section -- this is what
  makes it credible, not weak.

Output a numbered outline with section titles and one-line argument summaries."""

    outline = call_ai(outline_prompt, api_key, model, temp=0.4)

    pad("synthesis_02_outline", f"# Outline (thesis #{thesis_num})\n\n{outline}")
    print(outline[:1500])

    # ════════════════════════════════════════════════════════════════
    # 1.b.iii: WRITE THE BRIEF (section by section)
    # ════════════════════════════════════════════════════════════════
    # Now we write each section of the paper. The AI gets:
    #   - The chosen thesis
    #   - The full outline (so it knows where this section fits)
    #   - The evidence block (raw sources it can cite)
    #   - A system prompt with strict rules (no memory, no buzzwords)

    print(f"\n{'─'*50}\n1.b.iii: WRITE BRIEF\n{'─'*50}")

    # ── Parse section titles from the outline ───────────────────────
    # The outline looks like:
    #   1. Introduction: The productivity paradox of AI coding tools
    #   2. Evidence for productivity gains -- but from whom?
    #   3. The code quality gap: what the studies don't measure
    #   ...
    # We need to extract just the titles ("Introduction: The productivity
    # paradox of AI coding tools", etc.) from this numbered list.

    section_titles = []  # Will hold the extracted titles

    for raw_line in outline.split("\n"):  # Go through the outline line by line
        # Also strip Markdown heading/bold markers ("### 1. ...", "**1. ...**")
        # which models often add -- otherwise the pattern below never matches.
        line = raw_line.strip().lstrip("#* ")

        # re.match tries to match a pattern at the START of the string.
        # This pattern matches lines that start with a number or Roman numeral,
        # followed by a period or parenthesis, followed by text.
        #
        # r'^[\dIVXivx]+[\.\)]\s*(.+)' means:
        #   ^        = start of string
        #   [\dIVXivx]+  = one or more digits or Roman numeral characters
        #   [\.\)]   = a period or closing parenthesis
        #   \s*      = optional whitespace
        #   (.+)     = capture everything else (this is the title)
        m = re.match(r'^[\dIVXivx]+[\.\)]\s*(.+)', line)

        if m:
            # m.group(1) is the captured text (the title after the number)
            # .split("--")[0] takes only the part before an em-dash
            # .split(":")[0] takes only the part before a colon
            # This strips sub-descriptions: "Introduction: The paradox" → "Introduction"
            title = m.group(1).split("--")[0].split(":")[0].strip(" *")

            # Only keep titles that are reasonable length (not too short, not too long)
            if len(title) > 5 and len(title) < 80:
                section_titles.append(title)

    # Fallback: if parsing didn't find any titles, use generic ones
    # (this happens if the AI formatted the outline differently than expected)
    if not section_titles:
        section_titles = [
            "Introduction and Thesis",
            "Evidence Section 1",
            "Evidence Section 2",
            "Evidence Section 3",
            "Counterarguments and Limitations",
            "Conclusion",
        ]


    # ── Assign sources to sections (the "evidence map") ─────────────
    # Without this, every section got the same pile of evidence and a
    # cheap model leaned on the same 2-3 strong sources throughout. Now one
    # AI call reads all the cards and gives each section its OWN sources,
    # spreading the corpus across the paper. Each section is then written
    # from the full text of its assigned sources.
    print(f"\n{'─'*50}\nEVIDENCE MAP (sources per section)\n{'─'*50}")
    assignment = _assign_sources(section_titles, outline, evidence,
                                 set(usable), api_key, model) if usable else {}
    pad("synthesis_02b_evidence_map", "# Evidence map (sources per section)\n\n" + "\n".join(
        f"- {t}: {', '.join(f'[{i}]' for i in assignment.get(k, [])) or '(none -- uses all cards)'}"
        for k, t in enumerate(section_titles, 1)))
    assigned_all = {i for ids in assignment.values() for i in ids}
    if usable:
        print(f"   {len(assigned_all)} of {len(usable)} usable sources assigned to "
              f"{len(section_titles)} sections")

    # ── System prompt (rules the AI must follow) ────────────────────
    # The system prompt is like giving the AI its job description.
    # It runs before every section-writing call, setting the ground rules.
    #
    # ", ".join(...) creates a comma-separated string from a list:
    # ["a", "b", "c"] → "a, b, c"
    #
    # BANNED_PHRASES[:10] takes the first 10 banned phrases (to keep
    # the system prompt from getting too long).
    sys_prompt = (
        "You write academic research. STRICT RULES:\n"
        "- ONLY facts from the sources and evidence cards given. Tag every claim [n].\n"
        "- If evidence is thin, say so. NEVER fill from memory.\n"
        "- Note source language when relevant.\n"
        "- Draw on a RANGE of sources: support each point with several\n"
        "  different [n] sources where the evidence allows, and use the\n"
        "  non-English sources -- the corpus is multilingual on purpose.\n"
        "- One idea per sentence. Concrete. No jargon without definition.\n"
        "- NEVER use: " + ", ".join(f'"{p}"' for p in BANNED_PHRASES[:10]) + "\n"
        "- You are ARGUING a thesis, not summarizing a topic. Every paragraph\n"
        "  should advance the argument, not just present information."
    )

    # ── Write each section ──────────────────────────────────────────
    sections = []  # Will hold (title, content) pairs

    for k, title in enumerate(section_titles, 1):
        mine = assignment.get(k, [])
        print(f"    {title}  ({len(mine)} assigned sources)")

        # Full text of the sources assigned to THIS section
        assigned_text = "\n\n".join(
            f"[{i}][{by_idx[i].get('lang', '?')}] {by_idx[i].get('title', '')} "
            f"({by_idx[i].get('domain', '')})\n{by_idx[i].get('content', '')[:SOURCE_CHARS]}"
            for i in mine)

        # Ask the AI to write this specific section.
        # The section writer has never seen the thesis analysis, so we
        # must include the theses text itself -- "#2" alone means nothing.
        content = call_ai(
            f"Write this section of the paper.\n\n"
            f"THESIS: argue thesis #{thesis_num} from the analysis below.\n"
            f"THESES:\n{theses[:8000]}\n\n"
            f"SECTION: {title}\n"
            f"OUTLINE:\n{outline[:6000]}\n\n"  # Include outline for context
            + (f"ASSIGNED SOURCES -- build this section mainly on these, and use "
               f"most of them (full text):\n{assigned_text}\n\n" if assigned_text else "")
            + f"ALL EVIDENCE CARDS (one per source; you may cite any of these too):\n"
            f"{evidence[:CARDS_CHARS]}\n\n"
            f"400-600 words. Tag every claim [n]. Argue, don't summarize.",
            api_key, model, temp=0.5, system=sys_prompt,
        )

        # Store the section. If the AI call failed, use a placeholder.
        # "content or '*[Failed]*'" means: use content if it's not empty,
        # otherwise use the placeholder string.
        sections.append((title, content or "*[Failed]*"))

    # Save the raw (pre-cleaned) sections to the scratchpad
    # "\n\n---\n\n".join(...) puts "---" dividers between sections
    pad("synthesis_03_brief_raw",
        "\n\n---\n\n".join(f"## {t}\n\n{c}" for t, c in sections))

    # ════════════════════════════════════════════════════════════════
    # 1.b.iv: APA REFERENCES + VOICE CLEANING
    # ════════════════════════════════════════════════════════════════
    # Two final cleanup steps:
    #   1. Format all sources as APA 7th edition references
    #   2. Strip LLM-speak from the written sections

    print(f"\n{'─'*50}\nAPA + VOICE CLEAN\n{'─'*50}")

    # ── APA references ──────────────────────────────────────────────
    # Build a properly formatted reference list from our source metadata
    apa_text = _build_apa(args.corpus)
    pad("synthesis_04_apa", f"# References (APA 7th)\n\n{apa_text}")

    # ── Voice cleaning ──────────────────────────────────────────────
    # Scan each section for banned phrases (LLM-speak, buzzwords,
    # metaphors, hollow transitions) and remove them.
    #
    # Two-pass cleaning:
    #   Pass 1: Regex find-and-delete of known bad phrases
    #   Pass 2: If any phrases were found, ask the AI to rewrite
    #           the section cleanly (the regex leaves awkward gaps)

    cleaned = []  # Will hold the cleaned (title, content) pairs

    for title, content in sections:
        c = content   # Working copy (we'll modify this, not the original)
        flags = 0     # Count how many bad phrases we found

        # Check each banned phrase
        for phrase in BANNED_PHRASES:
            # re.compile creates a reusable pattern object.
            # re.escape makes sure special characters in the phrase
            # (like parentheses) are treated as literal text, not regex syntax.
            # re.IGNORECASE means "game-changer" matches "Game-Changer" too.
            pat = re.compile(re.escape(phrase), re.IGNORECASE)

            # Count how many times this phrase appears
            hits = len(pat.findall(c))

            if hits:
                flags += hits           # Add to total count
                c = pat.sub("", c)      # Remove all occurrences (replace with nothing)

        # Clean up the mess left behind (double spaces, orphaned punctuation).
        # r"[ \t]{2,}" matches 2+ spaces/tabs only -- NOT newlines, so the
        # paragraph breaks survive. Then squash 3+ newlines back to 2.
        c = re.sub(r"[ \t]{2,}", " ", c)
        c = re.sub(r"\n{3,}", "\n\n", c).strip()

        # If we found and removed phrases, the text probably reads awkwardly.
        # Ask the AI to rewrite it smoothly while keeping facts and citations.
        if flags > 0:
            rw = call_ai(
                f"Rewrite: remove buzzwords, keep facts and [n] tags. "
                f"Direct, concrete, argues a position.\n\n{c}",
                api_key, model, temp=0.3,  # Low temp = stay close to original
            )
            if rw:
                c = rw  # Use the rewritten version

            print(f"   {title}: {flags} phrases cleaned")

        cleaned.append((title, c))

    # ════════════════════════════════════════════════════════════════
    # ASSEMBLE THE FINAL REPORT
    # ════════════════════════════════════════════════════════════════
    # Combine all cleaned sections, metadata, and references into
    # one Markdown file.

    lines = []  # Will hold all lines of the final report

    # Header
    lines.append("# Research Brief\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**Pipeline:** explore → analyze → stats → synthesize")
    lines.append(f"**Thesis:** #{thesis_num} (evidence-derived)")

    # ── Coverage: how much of the corpus does the brief actually use? ─
    cited = sorted({int(n) for t, c in cleaned
                    for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", c)
                    for n in re.split(r"\s*,\s*", grp) if int(n) in by_idx})
    cited_langs = sorted({by_idx[i].get("lang", "?") for i in cited})
    coverage = (f"cites {len(cited)} of {len(usable) or len(corpus)} usable sources, "
                f"in {len(cited_langs)} languages ({', '.join(cited_langs)})")
    lines.append(f"**Evidence coverage:** {coverage}\n")
    lines.append("---\n")

    # Sections
    for t, c in cleaned:
        lines.append(f"## {t}\n\n{c}\n\n---\n")

    # References
    lines.append(f"## References\n\n{apa_text}")

    # Save to file
    # args.output is the user-specified filename, or None for auto-generated
    fn = args.output or f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))  # Join all lines with newlines

    # ── Done! ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SYNTHESIS COMPLETE")
    print(f"   {fn}")
    print(f"   Brief {coverage}")
    print("   Thesis was derived from evidence, not assumed.")
    print("   Run 'deepen' to find targeted support for weak areas.")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (used internally by run_synthesize)
# ════════════════════════════════════════════════════════════════════════
#
# Functions whose names start with _ are "private by convention" --
# they're meant to be used only inside this file, not called from
# outside. Python doesn't enforce this (unlike some other languages),
# but the underscore signals "internal use."

def _load_all_evidence(corpus_dir):
    """
    Load EVERYTHING from the scratchpad into one big text string.

    This is what the thesis generation AI reads. It includes:
      - Analysis results (sentiment scores, topic distributions)
      - Statistical test results (chi-squared, Mann-Whitney, etc.)
      - Red team critique (what's wrong with the research)
      - Bias audit (what's skewed)
      - Definitions and decomposition
      - Simulation results (if available)

    Each file is labeled with its name so the AI knows what it's reading.
    Files are truncated to keep the total under the AI's input limit.
    """
    parts = []  # Will hold text chunks from each file

    pad_dir = Path(corpus_dir)  # Convert string path to Path object

    # If the scratchpad folder doesn't exist, there's nothing to load
    if not pad_dir.exists():
        return ""

    # ── Analysis results (highest priority -- these ARE the findings) ──
    for fn in ["analysis_results.json", "corpus_analyzed.csv"]:
        p = pad_dir / fn  # Path / "filename" joins them (like os.path.join)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                # f"── {fn} ──\n" creates a label header so the AI knows
                # which file this text came from
                parts.append(f"── {fn} ──\n{f.read()[:3000]}\n")

    # ── Statistical reports ──────────────────────────────────────────
    # .glob("stats_*.md") finds all files matching "stats_*.md"
    for fn in sorted(pad_dir.glob("stats_*.md")):
        # fn.stem is the filename without extension: "stats_report.md" → "stats_report"
        # fn.read_text() reads the whole file as a string
        parts.append(f"── {fn.stem} ──\n{fn.read_text(encoding='utf-8')[:2000]}\n")

    # ── Red team and bias audit ──────────────────────────────────────
    for name in ["06_red_team", "065_bias", "01_definitions", "02_decomposition"]:
        p = pad_dir / f"{name}.md"
        if p.exists():
            parts.append(f"── {name} ──\n{p.read_text(encoding='utf-8')[:2000]}\n")

    # ── Simulation results (if the user ran simulate mode) ───────────
    sim = pad_dir / "simulation_results.json"
    if sim.exists():
        with open(sim) as f:
            parts.append(f"── simulation ──\n{f.read()[:2000]}\n")

    # Join all parts into one big string
    return "\n".join(parts)


def _load_corpus(corpus_dir):
    """Load corpus.json (list of sources with idx/title/content...), or []."""
    corpus_path = os.path.join(corpus_dir, "corpus.json")
    if not os.path.exists(corpus_path):
        return []
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def _assign_sources(section_titles, outline, cards_text, valid_ids, api_key, model):
    """
    Ask the AI which sources belong in which section.

    Returns {section number (1-based): [source idx, ...]}, keeping only
    real source numbers, at most MAX_SOURCES_PER_SECTION each. Sections the
    AI skipped (or a failed call) get no entry -- they're then written from
    the evidence cards alone.
    """
    numbered = "\n".join(f"SECTION {k}: {t}" for k, t in enumerate(section_titles, 1))
    reply = call_ai(
        "You are planning which evidence goes into each section of a paper.\n\n"
        f"OUTLINE (for each section's argument):\n{outline[:6000]}\n\n"
        f"SECTIONS:\n{numbered}\n\n"
        f"EVIDENCE CARDS (one per source, [n] = source number):\n{cards_text[:THESIS_CONTEXT_CHARS]}\n\n"
        f"For EACH section, list the 6-{MAX_SOURCES_PER_SECTION} sources that best support, "
        "illustrate or complicate its argument. Spread the evidence: use as many "
        "DIFFERENT relevant sources across the paper as you can (avoid giving every "
        "section the same few), and include non-English sources where relevant.\n"
        "Output exactly one line per section: 'SECTION <k>: n, n, n'",
        api_key, model, temp=0.2,
    )
    assignment = {}
    for k, ids in re.findall(r"SECTION\s*(\d+)\s*[:：]\s*(.+)", reply or ""):
        picked = []
        for n in re.findall(r"\d+", ids):
            if int(n) in valid_ids and int(n) not in picked:
                picked.append(int(n))
        if picked and 1 <= int(k) <= len(section_titles):
            assignment[int(k)] = picked[:MAX_SOURCES_PER_SECTION]

    # Spread the evidence: models tend to hand the same few strong sources
    # to every section. Keep each source in its first MAX_SECTIONS_PER_SOURCE
    # sections only, so later sections are built on other material.
    uses = defaultdict(int)
    for k in sorted(assignment):
        kept = []
        for i in assignment[k]:
            if uses[i] < MAX_SECTIONS_PER_SOURCE:
                uses[i] += 1
                kept.append(i)
        assignment[k] = kept
    return {k: ids for k, ids in assignment.items() if ids}


def _build_evidence_block(corpus_dir):
    """
    Build a tagged evidence block from the corpus for section writing.

    This creates a formatted text where each source has a number tag [1],
    [2], etc. When the AI writes a section, it cites sources using these
    numbers: "According to a Microsoft study [4], productivity increased..."

    Sources are organized by quality tier so the AI can see which are
    academic, which are vendor-published, which are from forums, etc.
    """

    corpus_path = os.path.join(corpus_dir, "corpus.json")

    # If the corpus doesn't exist, return empty string
    if not os.path.exists(corpus_path):
        return ""

    # Load the corpus (list of sources with their content)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)  # JSON → Python list of dictionaries

    evidence = "# EVIDENCE -- tag claims [n]\n\n"

    # Group sources by quality tier using defaultdict(list).
    # After this loop, by_q["academic"] is a list of all academic sources,
    # by_q["journalism"] is all journalism sources, etc.
    by_q = defaultdict(list)
    for item in corpus:
        by_q[item.get("quality", "unclassified")].append(item)

    # Add sources to the evidence block, organized by quality.
    # We process quality tiers in a specific order: academic first
    # (most authoritative), then down to unclassified.
    # Tiers not in this list (e.g. "SEO risk", "government") come after
    # it instead of being silently left out of the evidence.
    order = ["academic", "intl institution", "government", "vendor/primary",
             "consulting", "journalism", "experiential/forum",
             "experiential/blog", "unclassified"]
    order += sorted(q for q in by_q if q not in order)

    for quality in order:

        # by_q.get(quality, []) returns the list of sources for this tier,
        # or an empty list if there are none.
        # SOURCES_PER_TIER caps each tier so a huge "unclassified" tier
        # can't crowd out the academic and institutional sources.
        for item in by_q.get(quality, [])[:SOURCES_PER_TIER]:
            evidence += (
                # [3][ja] means source #3, in Japanese
                f"[{item.get('idx', '?')}][{item.get('lang', '?')}] "
                f"{item['title']} ({item['domain']})\n"
                # Indent the content preview for readability
                f"    {item['content'][:500]}\n\n"
            )

    return evidence


def _build_apa(corpus_dir):
    """
    Generate APA 7th edition references from the corpus.

    APA format for a web source looks like:
    Author. (Year). Title. Site Name. Retrieved Month Day, Year, from URL

    Since we usually don't know the author (we scraped the pages),
    we use the title and domain as available fields. When the publication
    year is unknown, APA uses "n.d." (no date) -- NOT the current year.

    Non-English sources get a language note: [In Japanese].

    We read corpus.json (not sources_meta.json) so each reference number
    is the same "idx" the evidence block uses for its [n] tags. deepen
    appends new sources to corpus.json only, so numbering by the metadata
    file would leave the new [n] tags pointing at missing references.
    """

    corpus_path = os.path.join(corpus_dir, "corpus.json")

    if not os.path.exists(corpus_path):
        return ""

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)  # List of sources, each with its "idx"

    entries = []  # Will hold formatted APA entries

    for pos, s in enumerate(corpus, 1):
        i = s.get("idx", pos)  # Same number the evidence block cites
        url = s.get("url", "")

        # Check if this source is in a non-English language
        sl = s.get("lang", "")
        lang_note = ""
        if sl and sl not in ("en", "unknown"):
            # LANGUAGES.get(sl, {}).get("name", sl) means:
            # Look up the language code in our LANGUAGES dictionary.
            # If found, get its "name" field.
            # If not found, just use the code itself (e.g., "ja").
            lang_note = f" [In {LANGUAGES.get(sl, {}).get('name', sl)}]."

        # Build the APA entry.
        # *{domain}* makes the domain italic in Markdown.
        # datetime.now().strftime('%B %d, %Y') → "October 07, 2026"
        entries.append(
            f"[{i}] {s.get('title', '')}. ({s.get('year') or 'n.d.'}). "
            f"*{s.get('domain', '')}*.{lang_note} "
            f"Retrieved {datetime.now().strftime('%B %d, %Y')}, from {url}"
        )

    # Join all entries with double newlines between them
    return "\n\n".join(entries)
