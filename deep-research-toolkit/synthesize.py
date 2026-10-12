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
import sys   # Stop with an error code when a step can't run
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
    CONTENT_CHARS,         # How much text is stored per source
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

BRIEF_WORDS = 6000             # Target length of the whole brief (~22 double-spaced
                               # pages). Split across sections by how much evidence
                               # each has; sections stop early if evidence runs out.
THESIS_CONTEXT_CHARS = 120000  # Findings + ALL evidence cards for the thesis step
CARDS_CHARS = 100000           # Evidence cards shown when writing each section
SOURCE_CHARS = CONTENT_CHARS   # Text per source given to the writer: all of it
FACT_CHECK_CHARS = 400000      # Room for the full text of every source a section
                               # cites (~100k tokens), so the checker sees it all
MAX_SOURCES_PER_SECTION = 15   # Sources the evidence map assigns per section
FINDINGS_FILE_CHARS = 20000     # Longest findings file kept (normal ones: 2-10k)
SOURCES_PER_PART = 5           # Long sections are written in passes of ~5 sources

# Banned phrases that only open a sentence ("Moreover, X") -- removed in code
TRANSITIONS = [
    "it's important to note that", "it is important to note that",
    "it's worth noting that", "it is worth noting that",
    "furthermore", "moreover", "additionally", "in conclusion", "to summarize",
    "as we have seen", "as discussed above",
    "en effet", "par ailleurs", "en conclusion", "darüber hinaus", "des Weiteren",
    "zusammenfassend lässt sich sagen", "en conclusión", "em conclusão",
]
                               # (models write ~400-500 words per call, however
                               # long they're asked to write)
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
        sys.exit("No API key found -- run install.py to add one (or pass --key KEY)")

    model = args.model  # Which AI model to use

    print("=" * 60)
    print("MODE 1.b: SYNTHESIZE (thesis from evidence)")
    print("=" * 60)

    # ── Load all evidence from the scratchpad ───────────────────────
    # This reads every file that explore, analyze, and stats produced.
    # The AI will use this as its ONLY source of information.
    context = _load_all_evidence(args.corpus)

    if not context:
        sys.exit("No evidence found. Run explore → analyze → stats first.")

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

The paper is about {BRIEF_WORDS} words (~{BRIEF_WORDS // 275} double-spaced pages), for a business school capstone.
The audience is financial industry professionals, not technologists.

THESES:
{theses}

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
        "- ONLY facts from the sources and evidence cards given. Tag every claim [n]\n"
        "  with the 1-3 sources that actually say it -- never long lists of sources.\n"
        "- Results of THIS study's own analysis (its themes, sentiment scores,\n"
        "  statistical tests, source counts) come from the research findings:\n"
        "  say so in words ('our analysis of the corpus found...'), with NO [n]\n"
        "  tag -- [n] is only for what a source itself says -- and never\n"
        "  mention file names.\n"
        "- Report numbers with the scope the source gives them (who was\n"
        "  studied, what was measured) -- don't generalise them.\n"
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
    # Length: BRIEF_WORDS split by how many sources each section was given
    # (more evidence → longer section), within 350-1500 words. The writer
    # is told to stop early rather than pad -- extra length must come from
    # extra evidence, never from general knowledge.
    weights = {k: max(3, len(assignment.get(k, []))) for k in range(1, len(section_titles) + 1)}
    targets = {k: min(1500, max(350, round(BRIEF_WORDS * w / sum(weights.values()))))
               for k, w in weights.items()}
    # A section with no assigned sources (often the conclusion) has only the
    # cards to draw on -- give it the minimum, not a share it can't fill
    for k in targets:
        if not assignment.get(k):
            targets[k] = 350

    sections = []    # Will hold (title, content) pairs
    unfiltered = []  # Every writing pass as the model wrote it (saved for review)

    for k, title in enumerate(section_titles, 1):
        mine = assignment.get(k, [])
        print(f"    {title}  ({len(mine)} assigned sources, ~{targets[k]} words)")

        # Write the section in passes of ~SOURCES_PER_PART sources each, so
        # a section with 15 sources gets ~3x the words of one with 5 -- and
        # every assigned source is actually used. Each pass sees the text so
        # far and continues it.
        n_parts = max(1, round(len(mine) / SOURCES_PER_PART))
        parts = [mine[p::n_parts] for p in range(n_parts)] if mine else [[]]
        part_words = max(300, targets[k] // n_parts)
        content = ""
        # Earlier sections' text -- only for dropping repeated sentences in code.
        # (Putting it in the prompt made the model copy it back, and the
        # sections shrank to almost nothing.)
        earlier = "\n\n".join(f"## {t}\n{c}" for t, c in sections)

        for p, part_sources in enumerate(parts, 1):
            # Full text of the sources for THIS pass
            assigned_text = "\n\n".join(
                f"[{i}][{by_idx[i].get('lang', '?')}] {by_idx[i].get('title', '')} "
                f"({by_idx[i].get('domain', '')})\n{by_idx[i].get('content', '')[:SOURCE_CHARS]}"
                for i in part_sources)
            where = (f"This is part {p} of {n_parts} of the section. Continue the text "
                     f"so far without repeating it or adding a heading"
                     + ("; you may conclude the section." if p == n_parts else
                        "; don't conclude yet.") + f"\n\nSECTION TEXT SO FAR:\n{content}\n\n"
                     if n_parts > 1 else "")

            # The section writer has never seen the thesis analysis, so we
            # must include the theses text itself -- "#2" alone means nothing.
            part_args = (
                f"Write this section of the paper.\n\n"
                f"THESIS: argue thesis #{thesis_num} from the analysis below.\n"
                f"THESES:\n{theses}\n\n"
                f"SECTION: {title}\n"
                f"OUTLINE:\n{outline}\n\n"  # Include outline for context
                + where
                + (f"ASSIGNED SOURCES -- build this {'part' if n_parts > 1 else 'section'} "
                   f"mainly on these, and use most of them (full text):\n{assigned_text}\n\n"
                   if assigned_text else "")
                + f"ALL EVIDENCE CARDS (one per source; you may cite any of these too):\n"
                f"{evidence[:CARDS_CHARS]}\n\n"
                f"RESEARCH FINDINGS (this study's own analysis and statistics -- use only "
                f"the ones that bear on THIS section's point, with no [n]):\n{context}\n\n"
                f"Length: about {part_words} words -- but ONLY as long as the evidence "
                f"supports. Use the specific numbers, findings and examples in the "
                f"sources. If the evidence runs out first, stop: shorter is better "
                f"than padding or general knowledge. Every factual sentence must come "
                f"from the sources or the research findings, with numbers copied "
                f"exactly -- a fact checker will delete anything they don't support. "
                f"Tag every claim [n] with the 1-3 sources that actually say it -- "
                f"never a long list of sources in one bracket. "
                f"Argue, don't summarize.",
                api_key, model,
            )
            part_kwargs = {"temp": 0.3, "system": sys_prompt}  # low temp = stay close to sources
            part = call_ai(*part_args, **part_kwargs)
            if not part:   # Failed call (e.g. network or credit) -- retry once
                part = call_ai(*part_args, **part_kwargs)
            if not part:
                print(f"      part {p} of {n_parts} FAILED -- this section is incomplete")
            if part:
                unfiltered.append(f"## {title} -- pass {p} of {n_parts}\n\n{part}")
                written = _words(part)
                # Citation brackets listing 6+ sources (the writer once tagged
                # this study's own statistics with 309 of them) say nothing
                # about which source supports what -- drop them; the fact
                # check then re-cites the sentence properly or removes it
                part = _drop_citation_dumps(part)
                # Drop sentences already written in this or an earlier section
                part = _drop_repeated_sentences(part, earlier + "\n\n" + content)
                print(f"      pass {p} of {n_parts}: {written} words written, "
                      f"{_words(part)} kept", flush=True)
            content = (content + "\n\n" + part).strip() if part else content

        # Store the section. If the AI call failed, use a placeholder.
        # "content or '*[Failed]*'" means: use content if it's not empty,
        # otherwise use the placeholder string.
        sections.append((title, content or "*[Failed]*"))

    # Save the raw (pre-cleaned) sections to the scratchpad
    # "\n\n---\n\n".join(...) puts "---" dividers between sections
    pad("synthesis_03a_passes_unfiltered", "\n\n---\n\n".join(unfiltered))
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
    # Deleting a phrase from the middle of a sentence leaves broken grammar
    # ("Furthermore, a paradigm shift looms." -> ", a looms."), so:
    #   1. Sentence-opening transitions ("Moreover,") are removed in code --
    #      the sentence still reads fine without them.
    #   2. Each sentence containing any other banned phrase is rewritten by
    #      the AI, and the rewrite is applied only if it kept every number
    #      and [n] tag and dropped the phrase. Otherwise the sentence stays
    #      as written: a buzzword is better than a broken sentence.

    cleaned = []  # Will hold the cleaned (title, content) pairs

    for title, content in sections:
        c, n_trans = _drop_transitions(content)
        bad = _sentences_with_banned(c)
        n_fixed = 0
        if bad:
            reply = call_ai(
                "Rewrite each sentence below without the phrase(s) in brackets. "
                "Keep its meaning, every fact, number and [n] tag. For EACH one, "
                "output exactly two lines:\nORIGINAL: <the sentence, copied exactly>\n"
                "FIX: <the rewritten sentence>\n\n"
                + "\n".join(f"{sent}   [{', '.join(ph)}]" for sent, ph in bad),
                api_key, model, temp=0.3, max_tokens=4000,
            )
            wanted = {sent: ph for sent, ph in bad}
            for original, fix in re.findall(r"ORIGINAL:\s*(.+?)\s*\n\s*FIX:\s*(.+)", reply or ""):
                original, fix = original.strip().strip('"'), fix.strip().strip('"')
                if original in wanted and _keeps_facts(original, fix, wanted[original]):
                    c = c.replace(original, fix)
                    n_fixed += 1
        if n_trans or bad:
            print(f"   {title}: {n_trans} transitions removed, "
                  f"{n_fixed} of {len(bad)} sentences with buzzwords rewritten")
        cleaned.append((title, c))

    # ════════════════════════════════════════════════════════════════
    # 1.b.v: FACT CHECK (every claim against the sources it cites)
    # ════════════════════════════════════════════════════════════════
    # The last step before saving, so nothing rewrites the text after it.
    # For each section, one AI call compares every factual sentence with
    # the full text of the sources it cites and lists the ones that aren't
    # supported (or cite nothing); a second call fixes only those sentences
    # -- correct, re-cite or delete, never add. The log is saved so a human
    # can review exactly what was changed.

    # Remove file names the writer may still cite as if they were sources
    # ("[analysis_results.json]", "(stats_report.md)") -- [n] is for sources
    cleaned = [(t, _strip_file_refs(c)) for t, c in cleaned]

    print(f"\n{'─'*50}\nFACT CHECK (claims vs. cited sources)\n{'─'*50}")
    checked, check_log, n_flagged, not_checked = [], [], 0, []
    n_deleted = 0
    for title, c in cleaned:
        fixed, flagged = _fact_check(c, by_idx, cards, api_key, model, findings=context)
        if flagged is None:
            # Never report a failed check as a clean one
            not_checked.append(title)
            print(f"   {title}: NOT CHECKED -- the AI call failed (see the error above)")
            check_log.append(f"## {title}\n\n- NOT CHECKED: the fact-check call failed. "
                             "Re-run synthesize, or check this section by hand.")
        else:
            n_flagged += len(flagged)
            deleted = sum(1 for f in flagged if f.endswith("→ DELETED"))
            n_deleted += deleted
            print(f"   {title}: {len(flagged)} unsupported/uncited sentences -- "
                  f"{len(flagged) - deleted} corrected, {deleted} deleted")
            check_log.append(f"## {title}\n\n" + ("\n".join(f"- {f}" for f in flagged)
                                                   or "- All claims supported by their sources"))
        checked.append((title, fixed))
    # Corrections can make two rewordings of one claim identical -- keep the first
    cleaned, so_far = [], ""
    for title, c in checked:
        c = _drop_repeated_sentences(c, so_far)
        cleaned.append((title, c))
        so_far += "\n\n" + c
    pad("synthesis_05_fact_check",
        f"# Fact check: {n_flagged} sentences flagged -- {n_flagged - n_deleted} corrected, "
        f"{n_deleted} deleted"
        + (f" -- {len(not_checked)} section(s) NOT CHECKED" if not_checked else "") + "\n\n"
        "Each line was flagged as not supported by the source(s) it cites (UNSUPPORTED)\n"
        "or as stating a fact without a source (UNCITED), then corrected or removed.\n\n"
        + "\n\n".join(check_log))

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
    # Citation brackets listing more than 5 sources are "dumps", not
    # evidence for a claim -- they're counted separately, not as coverage
    groups = [re.split(r"\s*,\s*", g) for _, c in cleaned
              for g in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", c)]
    dumps = sum(len(g) > 5 for g in groups)
    cited = sorted({int(n) for g in groups if len(g) <= 5 for n in g if int(n) in by_idx})
    cited_langs = sorted({by_idx[i].get("lang", "?") for i in cited})
    coverage = (f"cites {len(cited)} of {len(usable) or len(corpus)} usable sources, "
                f"in {len(cited_langs)} languages ({', '.join(cited_langs)})"
                + (f"; {dumps} citation(s) listing 6+ sources not counted" if dumps else ""))
    lines.append(f"**Evidence coverage:** {coverage}")
    words = sum(len(re.findall(r"\w+", c)) for _, c in cleaned)
    check_note = (f"**WARNING: {len(not_checked)} section(s) NOT fact-checked** "
                  f"({', '.join(not_checked)}) -- re-run synthesize" if not_checked else
                  f"{n_flagged} unsupported or uncited sentences flagged -- "
                  f"{n_flagged - n_deleted} corrected, {n_deleted} deleted")
    lines.append(f"**Length:** {words} words | **Fact check:** {check_note} "
                 f"(see scratchpad/synthesis_05_fact_check.md)\n")
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
    print(f"   {words} words; fact check flagged {n_flagged} sentences: "
          f"{n_flagged - n_deleted} corrected, {n_deleted} deleted")
    if not_checked:
        print(f"   WARNING: {len(not_checked)} section(s) were NOT fact-checked "
              f"(AI calls failed) -- re-run synthesize before using this brief")
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
    """
    parts = []  # Will hold text chunks from each file

    pad_dir = Path(corpus_dir)  # Convert string path to Path object

    # If the scratchpad folder doesn't exist, there's nothing to load
    if not pad_dir.exists():
        return ""

    # These files are summaries (a few KB each), so they're read in full.
    # (They used to be cut at 2,000-3,000 characters each.)

    # ── Analysis results (highest priority -- these ARE the findings) ──
    # (corpus_analyzed.csv is no longer included: it's raw data, and the
    # evidence cards already give the AI every source properly.)
    p = pad_dir / "analysis_results.json"
    if p.exists():
        parts.append(f"── Corpus analysis results (sentiment, topics, themes) ──\n"
                     f"{p.read_text(encoding='utf-8')}\n")

    # ── Statistical reports ──────────────────────────────────────────
    # .glob("stats_*.md") finds all files matching "stats_*.md"
    for fn in sorted(pad_dir.glob("stats_*.md")):
        # fn.stem is the filename without extension: "stats_report.md" → "stats_report"
        parts.append(f"── Statistical tests ──\n{fn.read_text(encoding='utf-8')}\n")

    # ── Red team and bias audit ──────────────────────────────────────
    # Human labels, not file names: the writer quotes these labels, and
    # once cited "[analysis_results.json]" as if it were a source
    for name, label in [("06_red_team", "Red-team critique of the corpus"),
                        ("065_bias", "Bias audit of the corpus"),
                        ("01_definitions", "Key term definitions"),
                        ("02_decomposition", "Research questions by language")]:
        p = pad_dir / f"{name}.md"
        if p.exists():
            parts.append(f"── {label} ──\n{p.read_text(encoding='utf-8')}\n")

    # ── Simulation results (if the user ran simulate mode) ───────────
    p = pad_dir / "simulation_report.md"
    if p.exists():
        parts.append(f"── Coding-assistant simulation results ──\n{p.read_text(encoding='utf-8')}\n")

    # Join all parts into one big string
    # A findings file is a few KB. One that's far bigger is a model that got
    # stuck repeating itself (a 270 KB bias audit once filled every prompt)
    parts = [x if len(x) <= FINDINGS_FILE_CHARS else
             x[:FINDINGS_FILE_CHARS] + "\n[... cut: the rest of this file repeats itself]\n"
             for x in parts]
    return "\n".join(parts)


def _load_corpus(corpus_dir):
    """Load corpus.json (list of sources with idx/title/content...), or []."""
    corpus_path = os.path.join(corpus_dir, "corpus.json")
    if not os.path.exists(corpus_path):
        return []
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def _fact_check(text, by_idx, cards, api_key, model, findings=""):
    """
    Check every factual sentence in one section against the sources it
    cites -- and, for claims about this study's own results (themes,
    sentiment, statistical tests), against `findings` (the analysis and
    stats outputs). Returns (corrected text, list of flagged sentences),
    or (text, None) if the check itself failed -- the caller then reports
    the section as NOT CHECKED (never as "all supported").
    """
    cited = sorted({int(n) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
                    for n in re.split(r"\s*,\s*", grp) if int(n) in by_idx})
    # The full text of each cited source (plus its card, a handy summary)
    sources = "\n\n".join(
        f"[{i}] {by_idx[i].get('title', '')} ({by_idx[i].get('domain', '')})\n"
        f"CARD: {cards.get(i, '')}\nTEXT: {by_idx[i].get('content', '')[:SOURCE_CHARS]}"
        for i in cited)[:FACT_CHECK_CHARS]

    # The (long) sources come FIRST and the section LAST, clearly marked:
    # with the section first, a model once listed sentences from the
    # SOURCES as problems, repeating them ~2,000 times.
    report = call_ai(
        "You are a strict fact-checker for a research brief. Below are the full "
        "texts of the sources a section cites ([n] numbers), then the SECTION.\n\n"
        f"<sources>\n{sources}\n</sources>\n\n"
        f"<research_findings>\n{findings}\n</research_findings>\n\n"
        f"<section>\n{text}\n</section>\n\n"
        "Claims about THIS study's own results (its themes, sentiment scores, "
        "statistical tests, source counts) need no [n] but must match the "
        "research findings -- flag them as UNSUPPORTED if they don't, or if "
        "they cite a source [n] for the study's own results.\n"
        "A number stated with a broader scope than the source gives it (e.g. "
        "one study's sample presented as all developers) is UNSUPPORTED.\n"
        "Check EVERY sentence IN THE SECTION (only those) that states a fact, "
        "number, or claim about the world:\n"
        "- UNSUPPORTED: the cited source(s) don't say this (wrong number, "
        "overstated, or about something else)\n"
        "- UNCITED: it states a fact but has no [n] tag\n"
        "Sentences that only argue, connect or interpret, without new facts, are fine.\n"
        "Output one line per problem, each sentence once: 'UNSUPPORTED: <exact "
        "sentence from the section>' or 'UNCITED: <exact sentence from the "
        "section>'. If there are none, output: ALL SUPPORTED",
        api_key, model, temp=0, max_tokens=4000,
    )
    if not report:
        return text, None   # The call failed: this section was NOT checked

    # Keep only real sentences from THIS section, each once (guards against a
    # model quoting the sources, or repeating itself)
    flat = re.sub(r"\s+", " ", text)
    flagged = []
    for kind, raw_sentence in re.findall(r"(UNSUPPORTED|UNCITED)\s*:\s*(.+)", report or ""):
        sentence = raw_sentence.strip().strip('"')
        if (len(sentence) > 15 and re.sub(r"\s+", " ", sentence) in flat
                and f"{kind}: {sentence}" not in flagged):
            flagged.append(f"{kind}: {sentence}")
    if not flagged:
        return text, []

    # Ask for a correction per flagged sentence -- NOT a rewritten section:
    # asking for a revised section once returned the list of problems in
    # place of the text. Applying sentence-level fixes in code guarantees
    # that nothing else in the section can change.
    reply = call_ai(
        "Each sentence under PROBLEMS failed a fact check against the cited "
        "sources. For EACH one, output exactly two lines:\n"
        "ORIGINAL: <the sentence, copied exactly from the section>\n"
        "FIX: <the sentence corrected to say only what the source says, with the "
        "right [n] -- or the single word DELETE if no source supports it>\n"
        "Prefer correcting to deleting: fix the wrong number, word or scope, or cite "
        "the source that does say it. DELETE only if nothing in the sources or the "
        "research findings supports any version of the sentence. "
        "Never add new claims.\n\n"
        f"<sources>\n{sources}\n</sources>\n\n"
        f"<research_findings>\n{findings}\n</research_findings>\n\n"
        f"<section>\n{text}\n</section>\n\n"
        "PROBLEMS:\n" + "\n".join(flagged),
        api_key, model, temp=0, max_tokens=8000,
    )
    # Apply fixes only to flagged sentences (the model may "fix" others too),
    # everywhere they occur. A flagged sentence with no usable fix -- or every
    # one, if the fix call failed -- is deleted rather than kept: the check
    # found it unsupported (conservative by design).
    proposed = {}
    for raw_original, raw_fix in re.findall(r"ORIGINAL:\s*(.+?)\s*\n\s*FIX:\s*(.+)", reply or ""):
        proposed[raw_original.strip().strip('"')] = raw_fix.strip().strip('"')
    fixed, outcomes = text, []
    for f in flagged:
        original = f.split(": ", 1)[1]
        fix = proposed.get(original, "DELETE")
        fixed = fixed.replace(original, "" if fix.upper() == "DELETE" else fix)
        # Record what happened to it, so a human can audit the checker
        outcomes.append(f + ("\n  → DELETED" if fix.upper() == "DELETE" else f"\n  → CORRECTED: {fix}"))
    fixed = re.sub(r"[ \t]{2,}", " ", fixed)      # Tidy gaps left by deletions
    fixed = re.sub(r"^[ \t]+", "", fixed, flags=re.MULTILINE)
    return fixed, outcomes


def _banned_pattern(phrase):
    """Regex for a banned phrase as whole words, case-insensitive."""
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)


def _drop_transitions(text):
    """
    Remove banned transition words ("Moreover,", "Furthermore,", "En effet,")
    where they open a sentence, and capitalise what follows. Returns
    (text, number removed). Mid-sentence occurrences are left to the rewrite.
    """
    n = 0
    for phrase in TRANSITIONS:
        pat = re.compile(r"(^|(?<=[.!?:]\s)|(?<=\n))" + re.escape(phrase) + r"\s*,?\s+(\w)",
                         re.IGNORECASE | re.MULTILINE)
        text, k = pat.subn(lambda m: m.group(1) + m.group(2).upper(), text)
        n += k
    return text, n


def _sentences_with_banned(text):
    """[(sentence, [banned phrases in it])] for each sentence with any."""
    found = []
    for line in text.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", line):
            phrases = [p for p in BANNED_PHRASES if _banned_pattern(p).search(sent)]
            if phrases and sent.strip() and (sent.strip(), phrases) not in found:
                found.append((sent.strip(), phrases))
    return found


def _keeps_facts(original, fix, phrases):
    """True if `fix` kept every number and [n] tag and dropped the phrases."""
    def facts(t):
        return sorted(re.findall(r"\[[\d,\s]+\]|\d+(?:[.,]\d+)?", t))
    return (len(fix) > 0.4 * len(original) and facts(fix) == facts(original)
            and not any(_banned_pattern(p).search(fix) for p in phrases))


def _drop_citation_dumps(text, most=5):
    """Remove [n, n, ...] brackets that list more than `most` sources."""
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+){%d,}\]" % most, "", text)


def _drop_repeated_sentences(text, earlier):
    """
    Remove sentences of `text` that already appear in `earlier`, or earlier
    in `text` itself (ignoring [n] tags, case and punctuation) -- writing
    passes sometimes restate the thesis and headline numbers at the start of
    every section, and fact-check corrections can turn two rewordings of a
    claim into the same sentence.
    """
    def key(sentence):
        no_tags = re.sub(r"\[[\d,\s]+\]", "", sentence)
        return re.sub(r"\W+", " ", no_tags).strip().lower()

    seen = {key(x) for x in re.split(r"(?<=[.!?])\s+", earlier) if len(key(x)) > 40}
    lines = []
    for line in text.split("\n"):   # Line by line, so lists and headings keep their shape
        kept = []
        for sentence in re.split(r"(?<=[.!?])[ \t]+", line):
            k = key(sentence)
            if len(k) <= 40:
                kept.append(sentence)
            elif k not in seen:      # First time: keep it, and drop any later copy
                seen.add(k)
                kept.append(sentence)
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _strip_file_refs(text):
    """Drop bracketed/parenthesised file-name citations and bare file names."""
    text = re.sub(r"\s*[\[(][^\])\n]*\.(?:json|md|csv)[^\])\n]*[\])]", "", text)
    text = re.sub(r"\b(?:the\s+)?[\w-]+\.(?:json|md|csv)(?:\s+files?)?\b", "our analysis", text,
                  flags=re.I)
    # Capitalise it where it now starts a sentence
    text = re.sub(r"(^|[.!?]\s+)our analysis", r"\1Our analysis", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _words(text):
    """Number of words in a text."""
    return len(re.findall(r"\w+", text or ""))


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
        f"OUTLINE (for each section's argument):\n{outline}\n\n"
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
