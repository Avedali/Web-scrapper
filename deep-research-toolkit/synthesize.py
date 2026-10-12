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

import claims   # The claim ledger: checked findings the brief is written from
import verify   # The fact-check gate

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
# HOW THE BRIEF IS WRITTEN
# ════════════════════════════════════════════════════════════════════════
#
# Every source is read (as an evidence card) to find the thesis and outline.
# The brief itself is written from the CLAIM LEDGER (claims.py): every
# source's claims, each checked against a quote from the source, merged when
# several sources report the same finding, and given to ONE section each --
# so a finding is written once, and a number can't land on the wrong source.
#
# Length isn't set in advance: a section is as long as its findings need. The
# writer explains a hard finding fully and states a simple one briefly; it
# never pads. A longer brief needs more evidence (deepen), not more words.
# Everything written then goes through the fact-check gate (verify.py).

BRIEF_WORDS = 6000             # Only tells the outline how big the paper is
THESIS_CONTEXT_CHARS = 120000  # Findings + ALL evidence cards for the thesis step
UNCITED_MATCH = 0.75          # An uncited sentence this close to a finding gets its sources
NEAR_REPEAT = 0.92            # Sentences this alike in meaning count as a repeat
CLAIMS_PER_PASS = 12           # Findings one writing call turns into prose. Long
                               # sections take several passes, each from DIFFERENT
                               # findings -- no pass sees another's text (shown the
                               # text so far, the model copied it)
FINDINGS_FILE_CHARS = 20000    # Longest findings file kept (normal ones: 2-10k)
EVIDENCE_CHARS = 40000         # Fallback evidence block (if cards fail)
SOURCES_PER_TIER = 15          # Max sources per quality tier in that fallback

# Banned phrases that only open a sentence ("Moreover, X") -- removed in code
TRANSITIONS = [
    "it's important to note that", "it is important to note that",
    "it's worth noting that", "it is worth noting that",
    "furthermore", "moreover", "additionally", "in conclusion", "to summarize",
    "as we have seen", "as discussed above",
    "en effet", "par ailleurs", "en conclusion", "darüber hinaus", "des Weiteren",
    "zusammenfassend lässt sich sagen", "en conclusión", "em conclusão",
]


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

    # A verdict from an earlier run must never stand for this one
    stale = os.path.join(args.corpus, "brief_verdict.json")
    if os.path.exists(stale):
        os.remove(stale)

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


    # ── The claim ledger (claims.py) ────────────────────────────────
    print(f"\n{'─'*50}\nCLAIM LEDGER (every source's claims, checked)\n{'─'*50}")
    ledger_corpus = [src for src in corpus if src["idx"] in usable] or corpus
    cache = claims.extract_claims(ledger_corpus, api_key, model, args.corpus)
    stats = claims.check_claims(cache, by_idx, api_key, corpus_dir=args.corpus)
    findings = claims.merge_claims(cache, by_idx)
    assignment, unused = claims.assign_claims(findings, section_titles, outline)
    n_assigned = sum(len(v) for v in assignment.values())
    print(f"   {stats['ok']} claims confirmed by their quotes, {stats['corrected']} corrected "
          f"to match them, {stats['dropped']} dropped")
    print(f"   {len(findings)} distinct findings ({sum(len(f['sources']) > 1 for f in findings)} "
          f"backed by 2+ sources); {n_assigned} given to sections"
          + (f", {unused} more related than room" if unused else ""))
    if not n_assigned:
        sys.exit("No checked claims to write from -- the claim ledger is empty "
                 "(see scratchpad/claims.json). Re-run synthesize.")
    pad("synthesis_02b_ledger", "# Claim ledger: findings per section\n\n" + "\n\n".join(
        f"## {t}\n\n" + "\n".join(
            f"- {{{f['id']}}} {f['claim']} [sources: {', '.join(str(x) for x in f['sources'])}]"
            for f in assignment.get(t, [])) for t in section_titles))

    blocks = claims.outline_blocks(outline, section_titles)
    study_results = _study_results(args.corpus)
    findings_by_id = {f["id"]: f for f in findings}

    # ── System prompt (rules the writer must follow) ────────────────
    sys_prompt = (
        "You write academic research from a ledger of checked findings. RULES:\n"
        "- Every sentence that states a fact carries the ID of the finding it comes\n"
        "  from, in braces: {C12} -- one ID per brace, {C3}{C9} for two. Use ONLY the\n"
        "  findings given -- never facts from memory.\n"
        "- One finding per sentence, or say clearly which source says what ('GitHub\n"
        "  reports X {C3}, while a Veracode study found Y {C9}').\n"
        "- Say 'studies' or 'several sources' only for a finding with 2+ sources.\n"
        "- Keep each finding's numbers, hedges (up to, often, may) and scope (which\n"
        "  study, sample, company, country, year) exactly. Never widen one study's\n"
        "  sample into 'developers' or 'companies' in general.\n"
        "- Attribute vendors' claims about their own products ('GitHub reports ...').\n"
        "- This study's own results (THIS STUDY'S RESULTS below) carry no ID: say\n"
        "  'our analysis of the corpus found ...', and never mention file names.\n"
        "- Your own reasoning -- topic sentences, transitions, conclusions drawn from\n"
        "  the findings -- carries no ID and reads as reasoning ('This suggests ...').\n"
        "- Length follows the ideas: explain a hard finding fully (what was measured,\n"
        "  on whom, why it matters for the argument); state a simple one in a\n"
        "  sentence. No padding, and never make the same point twice.\n"
        "- Note a finding's source language when it matters: the corpus is\n"
        "  multilingual on purpose.\n"
        "- NEVER use: " + ", ".join(f'"{p}"' for p in BANNED_PHRASES[:10]) + "\n"
        "- You are ARGUING the thesis, not summarizing: every paragraph advances it."
    )

    def finding_lines(group):
        """The findings of one pass, as the writer sees them."""
        out = []
        for f in group:
            srcs = [by_idx[x] for x in f["sources"][:3] if x in by_idx]
            who = "; ".join(f"{x.get('quality', '?')}, {x.get('lang', '?')}: "
                            f"\"{x.get('title', '')[:70]}\" ({x.get('domain', '')})" for x in srcs)
            quote = next(iter(f["quotes"].values()), "")
            out.append(f"{{{f['id']}}} {f['claim']}\n    [{len(f['sources'])} source(s) -- {who}]"
                       f"\n    quote: \"{quote[:300]}\"")
        return "\n".join(out)

    # ── Write each section, pass by pass ────────────────────────────
    sections, unfiltered, coverage_log = [], [], []
    for k, title in enumerate(section_titles, 1):
        mine = assignment.get(title, [])
        groups = _claim_groups(mine, CLAIMS_PER_PASS)
        print(f"    {title}  ({len(mine)} findings, {len(groups)} pass(es))")
        pieces, used, skipped = [], set(), {}

        def write(group, instruction, p_label):
            """One writing call: prose from `group`; returns the text with SKIP lines removed."""
            prompt = (
                f"THESIS: argue thesis #{thesis_num} from this analysis:\n{theses}\n\n"
                f"OUTLINE OF THE PAPER:\n{outline}\n\n"
                f"SECTION: {title}\nWHAT THIS SECTION ARGUES:\n{blocks.get(title, title)}\n\n"
                f"{instruction}\n\n"
                f"FINDINGS TO USE (cite each with its ID in braces, e.g. {{C12}}):\n"
                f"{finding_lines(group)}\n\n"
                f"THIS STUDY'S RESULTS (no ID; use only what bears on this section):\n"
                f"{study_results}\n\n"
                "Write the prose now (no heading). Then, on separate lines at the very end, "
                "list any finding you could not fit as: SKIP: C12 -- reason"
            )
            text = call_ai(prompt, api_key, model, temp=0.3, system=sys_prompt)
            if not text:   # Failed call (e.g. network or credit) -- retry once
                text = call_ai(prompt, api_key, model, temp=0.3, system=sys_prompt)
            if not text:
                print(f"      {p_label} FAILED -- this section is incomplete")
                return ""
            unfiltered.append(f"## {title} -- {p_label}\n\n{text}")
            for m in re.finditer(r"^\W*SKIP\W*\{?C\s*(\d+)\}?\s*[-–—:]*\s*(.*)$", text, re.M | re.I):
                skipped["C" + m.group(1)] = m.group(2).strip() or "no reason given"
            text = re.sub(r"^\W*SKIP\W*\{?C\d+.*$", "", text, flags=re.M | re.I).strip()
            used.update(set(_finding_ids(text)) & {f["id"] for f in group})
            return text

        for p, group in enumerate(groups, 1):
            instruction = (f"This is part {p} of {len(groups)} of the section; the other parts "
                           f"use other findings. " if len(groups) > 1 else "")
            instruction += ("Open the section by stating its point in a sentence or two. " if p == 1
                            else "Don't introduce the section again or restate the thesis: continue "
                                 "its argument with these findings. ")
            if p == len(groups):
                instruction += "You may close with a short conclusion drawn from the section."
            piece = write(group, instruction, f"pass {p} of {len(groups)}")
            if piece:
                pieces.append(piece)

        # Coverage: a finding the writer neither used nor skipped (with a
        # reason) gets one more pass -- nothing is silently dropped
        missing = [f for f in mine if f["id"] not in used and f["id"] not in skipped]
        if missing and pieces:
            piece = write(missing, "These findings belong in this section but weren't used yet. "
                          "Write one or two paragraphs that work them into its argument; they "
                          "come at the end of the section, so don't introduce the section "
                          "again.", "coverage pass")
            if piece:
                pieces.append(piece)
        missing = [f["id"] for f in mine if f["id"] not in used and f["id"] not in skipped]

        content = _ids_to_citations("\n\n".join(pieces), findings_by_id)
        content, recited = _recite_numbers(content, mine, findings, by_idx)
        if recited:
            coverage_log.append(f"## {title} -- re-cited\n\n" + "\n".join(f"- {r}" for r in recited))
        content = _drop_repeated_sentences(content, "\n\n".join(c for _, c in sections))
        print(f"      used {len(used)} of {len(mine)} findings"
              + (f", {len(skipped)} skipped with a reason" if skipped else "")
              + (f", {len(missing)} not used" if missing else "")
              + (f", {len(recited)} re-cited" if recited else "")
              + f" -- {_words(content)} words")
        coverage_log.append(
            f"## {title}\n\nUsed {len(used)} of {len(mine)} findings.\n"
            + "".join(f"- SKIPPED {{{i}}}: {why}\n" for i, why in skipped.items())
            + "".join(f"- NOT USED {{{i}}}\n" for i in missing))
        sections.append((title, content or "*[Failed]*"))

    pad("synthesis_03a_passes_unfiltered", "\n\n---\n\n".join(unfiltered))
    pad("synthesis_03b_coverage", "# Findings used per section\n\n" + "\n".join(coverage_log))
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

    print(f"\n{'─'*50}\nFACT-CHECK GATE (verify.py)\n{'─'*50}")
    checked, gate_log, verdict = verify.gate(cleaned, by_idx, study_results, api_key)
    # Corrections can make two rewordings of one claim identical -- keep the first
    cleaned, so_far = [], ""
    for title, c in checked:
        c = _drop_repeated_sentences(c, so_far)
        cleaned.append((title, c))
        so_far += "\n\n" + c
    pad("synthesis_05_fact_check",
        f"# Fact-check gate: {verify.verdict_line(verdict)}\n\n"
        "Each line is a sentence the gate reviewed -- flagged by the rules in code or the\n"
        "local judge, or cited no source -- and what happened to it. The independent\n"
        "audit at the end re-checked a random sample against the sources' full text.\n\n"
        + "\n\n".join(gate_log))
    with open(os.path.join(args.corpus, "brief_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

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
    lines.append(f"**Length:** {words} words")
    lines.append(f"**Fact-check gate:** {verify.verdict_line(verdict)} "
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
    print(f"   {words} words")
    print(f"   Fact-check gate: {verify.verdict_line(verdict)}")
    if verdict["status"] != "PASS":
        print("   Read scratchpad/synthesis_05_fact_check.md before using this brief.")
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


def _study_results(corpus_dir):
    """
    This study's own results, short: the analysis results and the statistical
    tests (which say in words what each compared). The writer reports them
    with no [n]; the gate checks uncited numbers against them.
    """
    parts = []
    for name, label in [("analysis_results.json", "Corpus analysis results"),
                        ("stats_report.md", "Statistical tests")]:
        p = Path(corpus_dir) / name
        if p.exists():
            parts.append(f"── {label} ──\n{p.read_text(encoding='utf-8')[:FINDINGS_FILE_CHARS]}")
    return "\n\n".join(parts)


def _claim_groups(findings, size):
    """
    Split a section's findings into passes of about `size`, grouped by
    sub-topic (by meaning) so each pass writes about related findings; the
    pass with the section's most relevant finding comes first.
    """
    if len(findings) <= size:
        return [findings] if findings else []
    import math
    n = math.ceil(len(findings) / size)
    model = verify._match_model()
    if model is None:
        return [findings[i::n] for i in range(n)]
    from sklearn.cluster import KMeans
    emb = model.encode([f["claim"] for f in findings])
    labels = KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(emb)
    groups = {}
    for f, label in zip(findings, labels):   # findings come most relevant first
        groups.setdefault(label, []).append(f)
    out = []
    for g in groups.values():                # A lopsided group is split in two
        out += [g[:len(g) // 2], g[len(g) // 2:]] if len(g) > 1.5 * size else [g]
    return out


def _finding_ids(text):
    """
    Every finding ID in a text, however the writer bracketed it: {C12},
    {C12, C14}, {C12}{C14}, [C12], {C 12}.
    """
    ids = []
    for group in re.findall(r"[{\[(]\s*(C\s*\d+(?:(?:\s*[,;/&]\s*|\s+)(?:and\s+)?C?\s*\d+)*)\s*[}\])]", text, re.I):
        ids += ["C" + n for n in re.findall(r"\d+", group)]
    return ids


def _ids_to_citations(text, findings_by_id, most=3):
    """
    Turn finding IDs ({C12}, {C12, C14}, ...) into [n] source numbers,
    sentence by sentence: at most `most` sources per sentence, shared fairly
    between the findings it cites (strongest source of each first). Unknown
    IDs and other leftovers in braces ("{Simulation Results}") are removed.
    """
    id_group = re.compile(r"\s*[{\[(]\s*C\s*\d+(?:(?:\s*[,;/&]\s*|\s+)(?:and\s+)?C?\s*\d+)*\s*[}\])]", re.I)

    def one_sentence(sentence):
        known = [i for i in dict.fromkeys(_finding_ids(sentence)) if i in findings_by_id]
        share = max(1, most // max(1, len(known)))
        done = set()

        def replace(match):
            mine = [i for i in dict.fromkeys(_finding_ids(match.group(0))) if i in known and i not in done]
            done.update(mine)
            cites = list(dict.fromkeys(n for i in mine for n in findings_by_id[i]["sources"][:share]))
            return " [" + ", ".join(map(str, cites)) + "]" if cites else ""

        sentence = id_group.sub(replace, sentence)
        sentence = re.sub(r"\s*\{[^{}]*\}", "", sentence)   # Other leftovers in braces
        sentence = re.sub(r"\]\s*\[", ", ", sentence)        # "[3] [7]" → "[3, 7]"
        return re.sub(r"\s+([.,;:])", r"\1", sentence)

    lines = []
    for line in text.split("\n"):
        lines.append(" ".join(one_sentence(x) for x in re.split(r"(?<=[.!?])[ \t]+", line)))
    return "\n".join(lines)


def _recite_numbers(text, section_findings, all_findings, by_idx):
    """
    Fix a misattributed number before the gate sees it. The writer sometimes
    puts a true finding's number next to a neighbouring finding's ID (the
    METR "19% longer" result, cited to sources that never mention it). For
    each sentence whose numbers aren't in its cited sources, find the finding
    that states ALL of them and is closest in meaning -- this section's
    first -- and cite its sources instead. Returns (text, log lines).
    """
    model = verify._match_model()
    if model is None:
        return text, []
    from sentence_transformers import util
    pools = {}

    def numbers_of(idx):
        if idx not in pools:
            pools[idx] = verify.numbers(by_idx.get(idx, {}).get("content", ""))
        return pools[idx]

    log, out = [], []
    for line in text.split("\n"):
        fixed = []
        for sentence in re.split(r"(?<=[.!?])[ \t]+", line):
            cites = verify.citations(sentence)
            nums = verify.number_readings(sentence)
            pool = set().union(*(numbers_of(c) for c in cites)) if cites else set()
            plain = re.sub(r"\[[\d,\s]+\]", "", sentence)
            if cites and (not nums or all(r & pool for r in nums.values())):
                fixed.append(sentence)      # Cited, and its numbers are in its sources
                continue
            if not cites and len(re.findall(r"\w+", plain)) < 8:
                fixed.append(sentence)
                continue
            # Uncited: only re-cite a sentence that closely matches one of THIS
            # section's findings (the writer left off its ID); reasoning stays uncited
            searches = (section_findings, all_findings) if cites else (section_findings,)
            need = 0.5 if cites else UNCITED_MATCH
            best = None
            for candidates in searches:
                fitting = [f for f in candidates
                           if all(r & verify.numbers(f["claim"] + " " + " ".join(f["quotes"].values()))
                                  for r in nums.values())
                           and all(r & set().union(*(numbers_of(x) for x in f["sources"][:3]))
                                   for r in nums.values())]
                if fitting:
                    sims = util.cos_sim(model.encode(plain, convert_to_tensor=True),
                                        model.encode([f["claim"] for f in fitting], convert_to_tensor=True))[0]
                    i = int(sims.argmax())
                    if float(sims[i]) >= need:
                        best = fitting[i]
                        break
            if best:
                new = re.sub(r"\s*\[[\d,\s]+\]", "", sentence)
                new = re.sub(r"([.!?]?)$", " [" + ", ".join(map(str, best["sources"][:3])) + r"]\1", new, count=1)
                log.append(f"{sentence}\n  → {new}  (finding {best['id']}: {best['claim'][:120]})")
                sentence = new
            fixed.append(sentence)
        out.append(" ".join(fixed))
    return "\n".join(out), log


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


def _drop_repeated_sentences(text, earlier, same_meaning=NEAR_REPEAT):
    """
    Remove sentences of `text` already said in `earlier`, or earlier in `text`
    itself: word for word (ignoring [n] tags, case and punctuation), or -- when
    the sentence-matching model is installed -- in other words with the same
    meaning (similarity >= same_meaning). Writing passes reword the thesis and
    the same list of risks; fact-check corrections can turn two rewordings of
    a claim into the same sentence.
    """
    def key(sentence):
        no_tags = re.sub(r"\[[\d,\s]+\]", "", sentence)
        return re.sub(r"\W+", " ", no_tags).strip().lower()

    earlier_keys = [key(x) for x in re.split(r"(?<=[.!?])\s+", earlier) if len(key(x)) > 40]
    seen = set(earlier_keys)
    model = verify._match_model() if same_meaning else None
    kept_emb = []
    if model is not None and earlier_keys:
        kept_emb = list(model.encode(earlier_keys, convert_to_tensor=True))

    def said_before(k):
        if k in seen:
            return True
        if model is None:
            return False
        import torch
        from sentence_transformers import util
        e = model.encode(k, convert_to_tensor=True)
        if kept_emb and float(util.cos_sim(e, torch.stack(kept_emb)).max()) >= same_meaning:
            return True
        kept_emb.append(e)
        return False

    lines = []
    for line in text.split("\n"):   # Line by line, so lists and headings keep their shape
        kept = []
        for sentence in re.split(r"(?<=[.!?])[ \t]+", line):
            k = key(sentence)
            if len(k) <= 40:
                kept.append(sentence)
            elif not said_before(k):   # First time: keep it, and drop any later copy
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
