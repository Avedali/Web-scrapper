"""
synthesize.py: Mode 1.b, the research brief.
=============================================
Run: python agent.py synthesize
Or:  launcher.py → option 4

Reads everything explore, analyze and stats saved, and writes a research
brief organized by finding. It reports what the evidence shows and does not
argue for a position:

  Bottom line       3-5 sentences that answer the research question
  Key findings      one per theme, each with its evidence counted in code
  Findings by theme most of the brief: what was measured, by whom, on whom,
                    where sources agree and where they conflict
  Disagreements     findings that contradict each other, and why
  Gaps              what this evidence cannot tell the reader
  Business stakes   what is at play for the reader (cost, risk, competition,
                    people, regulation): the evidence, then the authors' view
  Method            how the brief was made (written by code, with no AI call)

Every fact comes from the claim ledger (claims.py) and every sentence goes
through the fact-check gate (verify.py). Run this after explore, analyze and stats.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # Read/write structured data files
import sys   # Stop with an error code when a step can't run
import os    # File/folder operations
import re    # Pattern matching (used to parse outlines and clean text)

from datetime import datetime       # Timestamps for the final report
from collections import defaultdict # Dictionary that fills in a default for a missing key
from pathlib import Path            # Object-based file path handling

from explore import DEFAULT_TOPIC   # The research topic, if none was given
import claims   # The claim ledger: checked findings the brief is written from
import verify   # The fact-check gate

# From core.py, the shared helper module:
from core import (
    call_ai,               # Send prompts to the AI model
    pad,                   # Save work to the scratchpad
    build_evidence_cards,  # English summary per source, 1-6 sentences
    NO_EVIDENCE,           # Card text for sources with nothing usable
    apa_reference,         # One source as an APA reference
    translate_titles,      # English titles of non-English sources, for APA
    load_title_translations,  # The same, read from their cache
    BANNED_PHRASES,        # LLM-speak phrases to remove (used by _plain_words)
    source_label,          # Plain name of a source type, for readers
)


# ════════════════════════════════════════════════════════════════════════
# HOW THE BRIEF IS WRITTEN
# ════════════════════════════════════════════════════════════════════════
#
# The framing step reads one evidence card per source to pick the research
# question and the themes. The brief itself is written from the claim ledger
# (claims.py): each source's claims, each checked against a quote from that
# source. When several sources report the same finding they become one
# finding, and each finding goes to one theme. So a finding is written once,
# and a number stays with the source that reported it.
#
# No word count is set. A section is as long as its findings need: the writer
# explains a hard finding in full and states a simple one in a sentence. To
# get a longer brief, collect more evidence with deepen instead of asking for
# more words.
# Everything written goes through the fact-check gate (verify.py).

AUDIENCE = "financial industry professionals"   # Who the business stakes are written for
FRAMING_CONTEXT_CHARS = 600000  # Findings + all evidence cards shown to the framing step (~150k tokens)
UNCITED_MATCH = 0.75          # An uncited sentence this close to a finding gets its sources
NEAR_REPEAT = 0.92            # Sentences this alike in meaning count as a repeat
CLAIMS_PER_PASS = 12           # Findings per writing call. A long section takes
                               # several calls, each with different findings. No call
                               # sees another call's text: when it did, the model
                               # copied it.
FINDINGS_FILE_CHARS = 20000    # Longest findings file kept (normal ones: 2-10k)
EVIDENCE_CHARS = 40000         # Fallback evidence block (if cards fail)
SOURCES_PER_TIER = 15          # Max sources per quality tier in that fallback

# Banned phrases that only start a sentence ("Moreover, X"). Code removes them.
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
    Write the research brief. Steps:
    framing (question and themes) → claim ledger → theme sections →
    disagreements → gaps → fact-check gate → bottom line, key findings and
    business stakes (checked too) → method → save.
    """

    # ── Setup ───────────────────────────────────────────────────────
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("No API key found. Run install.py to add one (or pass --key KEY)")
    model = args.model  # Which AI model writes the brief
    topic = getattr(args, "topic", None) or DEFAULT_TOPIC

    # Delete the verdict of an earlier run, so it can't be mistaken for this one's
    stale = os.path.join(args.corpus, "brief_verdict.json")
    if os.path.exists(stale):
        os.remove(stale)

    print("=" * 60)
    print("MODE 1.b: SYNTHESIZE (research brief from the evidence)")
    print("=" * 60)

    # ── Load what explore, analyze and stats saved ──────────────────
    context = _load_all_evidence(args.corpus)
    if not context:
        sys.exit("No evidence found. Run explore → analyze → stats first.")
    print(f"   Loaded {len(context)} chars of evidence from scratchpad/")

    # ── One evidence card per source ────────────────────────────────
    # A card is a 1-6 sentence English summary of one source (see
    # core.build_evidence_cards). All cards fit in one prompt, so the
    # framing step sees every source, in every language. Analyze already
    # built them, so this step reads them from the cache with no new AI calls.
    corpus = _load_corpus(args.corpus)
    by_idx = {s["idx"]: s for s in corpus}
    cards = build_evidence_cards(corpus, api_key, model, args.corpus) if corpus else {}
    usable = {i: c for i, c in cards.items() if not c.startswith(NO_EVIDENCE)}
    if usable:
        evidence = "\n".join(
            f"[{i}][{by_idx[i].get('lang', '?')}][{by_idx[i].get('quality', '?')}] {c}"
            for i, c in sorted(usable.items()) if i in by_idx)
        print(f"   Read all {len(corpus)} sources: {len(usable)} evidence cards "
              f"({len(cards) - len(usable)} sources had nothing usable)")
    else:
        # No cards (the AI calls failed): use the first lines of each source
        evidence = _build_evidence_block(args.corpus)[:EVIDENCE_CHARS]
        print("   Evidence cards unavailable, using source snippets instead")
    pad("synthesis_00_evidence_cards",
        f"# Evidence cards (one per source, cite as [n])\n\n{evidence}")

    # ════════════════════════════════════════════════════════════════
    # 1. FRAMING: the research question and the themes that answer it
    # ════════════════════════════════════════════════════════════════
    # The brief is organized by theme. Each theme is a sub-question the
    # evidence can answer, such as "Does AI make developers faster?". A theme
    # is never a position to argue. The AI picks 5-7 themes after reading
    # every card and the study's own results.
    print(f"\n{'─'*50}\n1. FRAMING (research question and themes)\n{'─'*50}")
    question, themes, framing = _frame_brief(topic, context, evidence, api_key, model)
    titles = [t for t, _ in themes]
    pad("synthesis_01_framing", f"# Research question and themes\n\n{framing}")
    print(f"   Question: {question}")
    for t, q in themes:
        print(f"   - {t}: {q}")

    # ════════════════════════════════════════════════════════════════
    # 2. THE CLAIM LEDGER (claims.py)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n2. CLAIM LEDGER (every source's claims, checked)\n{'─'*50}")
    ledger_corpus = [src for src in corpus if src["idx"] in usable] or corpus
    cache = claims.extract_claims(ledger_corpus, api_key, model, args.corpus)
    stats = claims.check_claims(cache, by_idx, api_key, corpus_dir=args.corpus)
    findings = claims.merge_claims(cache, by_idx)
    theme_text = "\n".join(f"{t}\n{q}" for t, q in themes)
    assignment, unused = claims.assign_claims(findings, titles, theme_text)
    n_assigned = sum(len(v) for v in assignment.values())
    print(f"   {stats['ok']} claims confirmed by their quotes, {stats['corrected']} corrected "
          f"to match them, {stats['dropped']} dropped")
    print(f"   {len(findings)} distinct findings ({sum(len(f['sources']) > 1 for f in findings)} "
          f"backed by 2+ sources); {n_assigned} given to themes"
          + (f", {unused} more were related but over the limit" if unused else ""))
    if not n_assigned:
        sys.exit("No checked claims to write from: the claim ledger is empty "
                 "(see scratchpad/claims.json). Re-run synthesize.")
    pad("synthesis_02b_ledger", "# Claim ledger: findings per theme\n\n" + "\n\n".join(
        f"## {t}\n\n" + "\n".join(
            f"- {{{f['id']}}} {f['claim']} [sources: {', '.join(str(x) for x in f['sources'])}]"
            for f in assignment.get(t, [])) for t in titles))

    study_results = _study_results(args.corpus)
    profile = _corpus_profile(corpus, usable, by_idx)
    findings_by_id = {f["id"]: f for f in findings}
    sys_prompt = _writer_rules()

    def finding_lines(group):
        """The findings of one writing call, as the writer sees them."""
        out = []
        for f in group:
            srcs = [by_idx[x] for x in f["sources"][:3] if x in by_idx]
            who = "; ".join(f"{x.get('quality', '?')}, {x.get('lang', '?')}: "
                            f"\"{x.get('title', '')[:70]}\" ({x.get('domain', '')})" for x in srcs)
            quote = next(iter(f["quotes"].values()), "")
            out.append(f"{{{f['id']}}} {f['claim']}\n    [{len(f['sources'])} source(s): {who}]"
                       f"\n    quote: \"{quote[:300]}\"")
        return "\n".join(out)

    unfiltered, coverage_log = [], []

    def write_section(title, purpose, mine, passes, written_so_far):
        """
        Write one section from its findings, in passes of CLAIMS_PER_PASS.
        `purpose` says what the section is for. `passes(p, n)` gives the
        instruction for pass p of n. Returns the text with [n] citations.
        """
        groups = _claim_groups(mine, CLAIMS_PER_PASS)
        pieces, used, skipped = [], set(), {}

        def write(group, instruction, label):
            """One writing call. Returns its text without the SKIP lines."""
            prompt = (
                f"RESEARCH QUESTION OF THE BRIEF: {question}\n\n"
                f"SECTION: {title}\nWHAT THIS SECTION IS FOR: {purpose}\n\n"
                f"{instruction}\n\n"
                f"FINDINGS TO USE (cite each with its ID in braces, e.g. {{C12}}):\n"
                f"{finding_lines(group)}\n\n"
                f"THIS STUDY'S OWN RESULTS (no ID; use only what bears on this section):\n"
                f"{study_results}\n\n"
                "Write the prose now (no heading). Then, on separate lines at the very end, "
                "list any finding you could not fit as: SKIP: C12 - reason"
            )
            text = call_ai(prompt, api_key, model, temp=0.3, system=sys_prompt)
            if not text:   # The call failed (network or credit): try once more
                text = call_ai(prompt, api_key, model, temp=0.3, system=sys_prompt)
            if not text:
                print(f"      {label} FAILED: this section is incomplete")
                return ""
            unfiltered.append(f"## {title} ({label})\n\n{text}")
            for m in re.finditer(r"^\W*SKIP\W*\{?C\s*(\d+)\}?\s*[-–—:]*\s*(.*)$", text, re.M | re.I):
                skipped["C" + m.group(1)] = m.group(2).strip() or "no reason given"
            text = re.sub(r"^\W*SKIP\W*\{?C\d+.*$", "", text, flags=re.M | re.I).strip()
            used.update(set(_finding_ids(text)) & {f["id"] for f in group})
            return text

        for p, group in enumerate(groups, 1):
            piece = write(group, passes(p, len(groups)), f"pass {p} of {len(groups)}")
            if piece:
                pieces.append(piece)

        # A finding the writer neither used nor skipped with a reason gets
        # one more pass, so no finding disappears without a note in the log
        missing = [f for f in mine if f["id"] not in used and f["id"] not in skipped]
        if missing and pieces:
            piece = write(missing, "These findings belong in this section but were not used "
                          "yet. Write one or two paragraphs that add them. They come at the "
                          "end of the section, so don't introduce the section again.",
                          "coverage pass")
            if piece:
                pieces.append(piece)
        missing = [f["id"] for f in mine if f["id"] not in used and f["id"] not in skipped]

        content = _ids_to_citations("\n\n".join(pieces), findings_by_id)
        content, recited = _recite_numbers(content, mine, findings, by_idx)
        if recited:
            coverage_log.append(f"## {title} (re-cited)\n\n" + "\n".join(f"- {r}" for r in recited))
        content = _drop_repeated_sentences(content, written_so_far)
        print(f"      used {len(used)} of {len(mine)} findings"
              + (f", {len(skipped)} skipped with a reason" if skipped else "")
              + (f", {len(missing)} not used" if missing else "")
              + (f", {len(recited)} re-cited" if recited else "")
              + f": {_words(content)} words")
        coverage_log.append(
            f"## {title}\n\nUsed {len(used)} of {len(mine)} findings.\n"
            + "".join(f"- SKIPPED {{{i}}}: {why}\n" for i, why in skipped.items())
            + "".join(f"- NOT USED {{{i}}}\n" for i in missing))
        return content

    # ════════════════════════════════════════════════════════════════
    # 3. FINDINGS BY THEME (most of the brief)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n3. FINDINGS BY THEME\n{'─'*50}")

    def theme_pass(p, n):
        """The instruction for pass p of n of a theme section."""
        text = (f"This is part {p} of {n} of the section; the other parts use other "
                f"findings. " if n > 1 else "")
        text += ("Open with one or two sentences on what the evidence on this question "
                 "shows overall. " if p == 1 else
                 "Continue the analysis with these findings. Don't introduce the section "
                 "again. ")
        if p == n:
            text += "End with what stays uncertain on this question, if the findings show it."
        return text

    theme_sections = []
    for title, sub_question in themes:
        mine = assignment.get(title, [])
        print(f"    {title}  ({len(mine)} findings)")
        written = "\n\n".join(c for _, c in theme_sections)
        content = write_section(title, f"answer the question: {sub_question}", mine,
                                theme_pass, written)
        theme_sections.append((title, content or "*[Failed]*"))

    # ════════════════════════════════════════════════════════════════
    # 4. WHERE THE EVIDENCE DISAGREES
    # ════════════════════════════════════════════════════════════════
    # Pairs of findings on the same subject that contradict each other,
    # found by the local judge (claims.find_disagreements). The brief shows
    # them in a section of their own, where an essay would put them under
    # counterarguments.
    print(f"\n{'─'*50}\n4. WHERE THE EVIDENCE DISAGREES\n{'─'*50}")
    in_brief = [f for t in titles for f in assignment.get(t, [])]
    pairs = claims.find_disagreements(in_brief)
    print(f"   {len(pairs)} pairs of findings that contradict each other")
    disagreements = ""
    if pairs:
        pair_findings = list({f["id"]: f for a, b in pairs for f in (a, b)}.values())
        pair_list = "\n".join(f"- {{{a['id']}}} versus {{{b['id']}}}" for a, b in pairs)
        disagreements = write_section(
            "Where the evidence disagrees",
            "show where findings contradict each other and explain each disagreement. "
            "These are the pairs that disagree:\n" + pair_list,
            pair_findings,
            lambda p, n: ("For each pair, say what each side found, who measured it, how "
                          "and on whom. Explain a difference only from what the findings say "
                          "(different measures, samples, vendors versus independent "
                          "researchers); if they don't say why, say the reason is not known."),
            "\n\n".join(c for _, c in theme_sections))

    # ════════════════════════════════════════════════════════════════
    # 5. GAPS IN THE EVIDENCE
    # ════════════════════════════════════════════════════════════════
    # Written from explore's red-team critique and bias audit plus the
    # corpus profile counted in code (languages, source types).
    print(f"\n{'─'*50}\n5. GAPS IN THE EVIDENCE\n{'─'*50}")
    critique = _scratch_file(args.corpus, "06_red_team") + "\n\n" + _scratch_file(args.corpus, "065_bias")
    gaps = call_ai(
        f"RESEARCH QUESTION: {question}\n\n"
        f"CORPUS PROFILE (counted in code):\n{profile}\n\n"
        f"CRITIQUE OF THE CORPUS (from this study's red team and bias audit):\n"
        f"{critique[:FINDINGS_FILE_CHARS]}\n\n"
        "Write the section 'Gaps in the evidence' of a research brief: what this body of "
        "evidence cannot tell the reader, and why (missing populations, regions, methods, "
        "time periods, source types; over-reliance on vendors or on one language). Use only "
        "the profile and critique above, and take every number from them. 2-4 paragraphs, "
        "plain words, no dashes as punctuation, no headings, no citations.",
        api_key, model, temp=0.3) or ""
    gaps = _strip_file_refs(gaps.strip())
    print(f"   {_words(gaps)} words")

    pad("synthesis_03a_passes_unfiltered", "\n\n---\n\n".join(unfiltered))
    pad("synthesis_03b_coverage", "# Findings used per section\n\n" + "\n".join(coverage_log))

    # ════════════════════════════════════════════════════════════════
    # 6. REFERENCES AND PLAIN WORDS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n6. REFERENCES AND PLAIN WORDS\n{'─'*50}")
    translate_titles(corpus, api_key, model, args.corpus)   # "Title [English title]" for APA
    apa_text = _build_apa(args.corpus)
    pad("synthesis_04_apa", f"# References (APA 7th)\n\n{apa_text}")
    body = theme_sections + [("Where the evidence disagrees", disagreements),
                             ("Gaps in the evidence", gaps)]
    body = [(t, _plain_words(c, t, api_key, model)) for t, c in body if c]
    body = [(t, _strip_file_refs(c)) for t, c in body]
    pad("synthesis_03_brief_raw", "\n\n---\n\n".join(f"## {t}\n\n{c}" for t, c in body))

    # ════════════════════════════════════════════════════════════════
    # 7. FACT-CHECK GATE (verify.py) on everything written so far
    # ════════════════════════════════════════════════════════════════
    # The gate checks numbers against the study's own results too, so it
    # gets the corpus profile and the critique the gaps section used.
    print(f"\n{'─'*50}\n7. FACT-CHECK GATE (verify.py)\n{'─'*50}")
    gate_facts = study_results + "\n\n" + profile + "\n\n" + critique[:FINDINGS_FILE_CHARS]
    checked, gate_log, verdict = verify.gate(body, by_idx, gate_facts, api_key)
    body = _drop_repeats_between(checked)
    checked_text = dict(body)

    # ════════════════════════════════════════════════════════════════
    # 8. BOTTOM LINE, KEY FINDINGS, BUSINESS STAKES (from the checked text)
    # ════════════════════════════════════════════════════════════════
    # Written last, from sections the gate already checked, so the summary
    # can only repeat what passed. The gate then checks these too.
    print(f"\n{'─'*50}\n8. BOTTOM LINE, KEY FINDINGS, BUSINESS STAKES\n{'─'*50}")
    bottom_line, key_findings = _summarize(question, themes, checked_text, api_key, model)
    stakes = _business_stakes(question, bottom_line, key_findings, checked_text, api_key, model)
    # The summary writer often drops the [n] citations it was told to keep.
    # Each uncited summary sentence borrows the citations of the checked
    # sentences it matches best (its own theme's section for a key finding,
    # any section for the bottom line), so the gate can check it.
    all_checked = "\n\n".join(checked_text.values())
    bottom_line = _cite_from_sections(bottom_line, all_checked)
    key_findings = [_cite_from_sections(k, checked_text.get(t, "")) if k else ""
                    for k, (t, _) in zip(key_findings, themes)]
    stakes = _cite_from_sections(stakes, all_checked)
    summary = [("Bottom line", bottom_line),
               ("Key findings", "\n".join(f"{n}. {k}" for n, k in enumerate(key_findings, 1) if k)),
               ("Business stakes and our view", stakes)]
    summary_checked, summary_log, summary_verdict = verify.gate(
        [(t, c) for t, c in summary if c], by_idx, gate_facts, api_key, run_audit=False)
    verdict = verify.merge_verdicts(verdict, summary_verdict)
    gate_log += summary_log
    summary_text = dict(summary_checked)

    pad("synthesis_05_fact_check",
        f"# Fact-check gate: {verify.verdict_line(verdict)}\n\n"
        "Each line is a sentence the gate reviewed (flagged by the rules in code or the\n"
        "local judge, or citing no source) and what happened to it. The independent\n"
        "audit at the end re-checked a random sample against the sources' full text.\n\n"
        + "\n\n".join(gate_log))
    with open(os.path.join(args.corpus, "brief_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    # ════════════════════════════════════════════════════════════════
    # 9. ASSEMBLE THE BRIEF
    # ════════════════════════════════════════════════════════════════
    all_text = "\n\n".join(list(checked_text.values()) + list(summary_text.values()))
    cited = sorted({n for n in verify.citations(all_text) if n in by_idx})
    cited_langs = sorted({by_idx[i].get("lang", "?") for i in cited})
    coverage = (f"cites {len(cited)} of {len(usable) or len(corpus)} usable sources, "
                f"in {len(cited_langs)} languages ({', '.join(cited_langs)})")
    method = _method_section(topic, corpus, usable, by_idx, stats, findings, n_assigned,
                             verdict, model, args.corpus)
    words = _words(all_text) + _words(method)

    lines = [f"# Research Brief: {question}\n",
             f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
             f"**Evidence:** {coverage}  ",
             f"**Length:** {words} words  ",
             f"**Fact-check gate:** {verify.verdict_line(verdict)} "
             f"(details: scratchpad/synthesis_05_fact_check.md)\n",
             "---\n",
             f"## Bottom line\n\n{summary_text.get('Bottom line', '*[Not written]*')}\n"]

    # Key findings, each followed by the evidence of its theme, counted in code
    key_text = summary_text.get("Key findings", "")
    found = {int(n): text.strip() for n, text in re.findall(r"^(\d+)\.[ \t]*(.*)$", key_text, re.M)}
    lines.append("## Key findings\n")
    shown = 0
    for n, (title, _) in enumerate(themes, 1):
        if found.get(n):   # A key finding the gate removed is skipped, so no empty item appears
            shown += 1
            lines.append(f"{shown}. {found[n]}  \n   *{title}. "
                         f"{_evidence_line(checked_text.get(title, ''), by_idx)}*")
    lines.append("")

    lines.append("## Findings by theme\n")
    for n, (title, sub_question) in enumerate(themes, 1):
        if title in checked_text:
            lines.append(f"### {n}. {title}\n\n*{sub_question} "
                         f"{_evidence_line(checked_text[title], by_idx)}*\n\n{checked_text[title]}\n")
    for title in ("Where the evidence disagrees", "Gaps in the evidence"):
        if checked_text.get(title):
            lines.append(f"## {title}\n\n{checked_text[title]}\n")
    if summary_text.get("Business stakes and our view"):
        # The opening line is written here, so readers know what is fact and what is view
        lines.append("## Business stakes and our view\n\n"
                     f"*For {AUDIENCE}. The stakes and the views are the authors' reading of the "
                     "evidence above; every fact is cited and was checked.*\n\n"
                     f"{summary_text['Business stakes and our view']}\n")
    lines.append(f"## Method\n\n{method}\n")
    lines.append(f"## References\n\n{apa_text}")

    # What the charts (viz) are drawn from: the question, each theme's key
    # finding and the types of sources behind it, and the findings used
    from collections import Counter
    with open(os.path.join(args.corpus, "brief_findings.json"), "w", encoding="utf-8") as f:
        json.dump({
            "question": question,
            "themes": [{
                "title": title,
                "question": sub_question,
                "key_finding": found.get(n, ""),
                "sources_by_type": dict(Counter(
                    _kind(by_idx[c]) for c in set(verify.citations(checked_text.get(title, "")))
                    if c in by_idx)),
            } for n, (title, sub_question) in enumerate(themes, 1)],
            "findings": [{"id": f["id"], "claim": f["claim"], "sources": f["sources"], "theme": t}
                         for t in titles for f in assignment.get(t, [])],
        }, f, indent=1, ensure_ascii=False)

    fn = args.output or f"brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print("SYNTHESIS COMPLETE")
    print(f"   {fn}")
    print(f"   Brief {coverage}")
    print(f"   {words} words")
    print(f"   Fact-check gate: {verify.verdict_line(verdict)}")
    if verdict["status"] != "PASS":
        print("   Read scratchpad/synthesis_05_fact_check.md before using this brief.")
    print("   Run 'deepen' to look for evidence on the gaps.")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (used internally by run_synthesize)
# ════════════════════════════════════════════════════════════════════════
#
# Functions whose names start with _ are "private by convention": they
# are meant to be used only inside this file. Python doesn't enforce this
# (some other languages do). The underscore tells other programmers that
# the function is for internal use.

def _load_all_evidence(corpus_dir):
    """
    Load all the findings files from the scratchpad into one text string.

    This is the text the AI reads when it frames the brief. It includes:
      - Analysis results (sentiment scores, topic distributions)
      - Statistical test results (chi-squared, Mann-Whitney, etc.)
      - Red team critique (what's wrong with the research)
      - Bias audit (what's skewed)
      - Definitions and decomposition
      - Simulation results (if available)

    Each file gets a label so the AI knows what it's reading.
    """
    parts = []  # Will hold text chunks from each file

    pad_dir = Path(corpus_dir)  # Convert string path to Path object

    # If the scratchpad folder doesn't exist, there's nothing to load
    if not pad_dir.exists():
        return ""

    # These files are summaries (a few KB each), so they're read in full.
    # (They used to be cut at 2,000-3,000 characters each.)

    # ── Analysis results (highest priority: these are the study's findings) ──
    # (corpus_analyzed.csv is no longer included: it's raw data, and the
    # evidence cards already give the AI a summary of every source.)
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
    # Readable labels are used instead of file names. The writer quotes these
    # labels, and once cited "[analysis_results.json]" as if it were a source.
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

    # Join all parts into one string.
    # A findings file is a few KB. A much bigger one comes from a model that got
    # stuck repeating itself (a 270 KB bias audit once filled every prompt), so
    # each file is cut at FINDINGS_FILE_CHARS.
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
    Return this study's own results in short form: the analysis results and
    the statistical tests (which say in words what each compared). The writer reports them
    with no [n]; the gate checks uncited numbers against them.
    """
    parts = []
    for name, label in [("analysis_results.json", "Corpus analysis results"),
                        ("stats_report.md", "Statistical tests")]:
        p = Path(corpus_dir) / name
        if p.exists():
            parts.append(f"── {label} ──\n{p.read_text(encoding='utf-8')[:FINDINGS_FILE_CHARS]}")
    return "\n\n".join(parts)


def _frame_brief(topic, context, evidence, api_key, model):
    """
    Ask the AI for the research question and 5-7 themes (sub-questions the
    evidence can answer). Returns (question, [(title, sub-question)], the
    AI's answer). If the answer can't be read, the themes come from
    analyze's topic model, or a fixed list as a last resort.
    """
    material = context + "\n\n" + evidence
    if len(material) > FRAMING_CONTEXT_CHARS:
        print(f"   Framing reads the first {FRAMING_CONTEXT_CHARS:,} of {len(material):,} characters "
              "of evidence (raise FRAMING_CONTEXT_CHARS to read all)")
    reply = call_ai(
        f"You are preparing a research brief on this topic: {topic}\n\n"
        "Below is everything the study found: its own analysis and statistics, then one "
        "evidence card per source.\n\n"
        "1. Write the research question the brief answers, in one sentence.\n"
        "2. Group the evidence into 5-7 THEMES that together answer it. A theme is a "
        "sub-question the evidence can answer (\"Do AI coding assistants make developers "
        "faster?\"), not a position. Cover what the evidence is about, including costs and "
        "risks, not only benefits. No theme about the study's own method.\n\n"
        "Answer in exactly this format and nothing else:\n"
        "QUESTION: <the research question>\n"
        "THEME 1: <short title, 2-6 words> | <the sub-question it answers>\n"
        "THEME 2: ...\n\n"
        f"EVIDENCE:\n{material[:FRAMING_CONTEXT_CHARS]}",
        api_key, model, temp=0.3) or ""
    m = re.search(r"^\W*QUESTION\W*:?\s*(.+)$", reply, re.M | re.I)
    question = m.group(1).strip() if m else topic
    themes = []
    for m in re.finditer(r"^\W*THEME\s*\d+\W*:?\s*(.+?)\s*\|\s*(.+)$", reply, re.M | re.I):
        title = re.sub(r"[*_#]", "", m.group(1)).strip()
        if title and title not in [t for t, _ in themes]:
            themes.append((title, m.group(2).strip()))
    if len(themes) < 3:
        print("   Could not read the themes from the AI's answer; using fallback themes")
        themes = _fallback_themes()
    return question, themes[:7], reply


def _fallback_themes():
    """Themes from analyze's topic model, or a fixed list if there's none."""
    p = Path("scratchpad") / "analysis_results.json"
    try:
        names = list(json.loads(p.read_text(encoding="utf-8")).get("themes", {}))[:6]
    except (OSError, ValueError):
        names = []
    if len(names) < 3:
        names = ["Productivity and speed", "Code quality and security", "Developers and skills",
                 "Adoption and business models", "Governance and regulation"]
    return [(n, f"What does the evidence say about {n.lower()}?") for n in names]


def _writer_rules():
    """The system prompt every writing call gets."""
    return (
        "You write the findings of a research brief from a ledger of checked findings. RULES:\n"
        "- Every sentence that states a fact carries the ID of the finding it comes from,\n"
        "  in braces: {C12}. One ID per brace: {C3}{C9} for two. Use ONLY the findings\n"
        "  given, never facts from memory.\n"
        "- One finding per sentence, or say clearly which source says what ('GitHub\n"
        "  reports X {C3}, while a Veracode study found Y {C9}').\n"
        "- Say 'studies' or 'several sources' only for a finding with 2+ sources.\n"
        "- Keep each finding's numbers, hedges (up to, often, may) and scope (which study,\n"
        "  sample, company, country, year) exactly. Never widen one study's sample into\n"
        "  'developers' or 'companies' in general.\n"
        "- Attribute vendors' claims about their own products ('GitHub reports ...').\n"
        "- This study's own results carry no ID: write 'our analysis of the corpus\n"
        "  found ...'. Never mention file names.\n"
        "- Analyze, don't argue for a position. For each point say what was measured, by\n"
        "  whom, on whom, and how strong the evidence is (one vendor blog, or several\n"
        "  independent studies). Say where sources agree, where they conflict, and what\n"
        "  the evidence cannot show.\n"
        "- Your own interpretation carries no ID and is marked as interpretation\n"
        "  ('Taken together, these findings suggest ...').\n"
        "- Length follows the ideas: explain a hard finding in full, state a simple one in\n"
        "  a sentence. No padding, and never make the same point twice.\n"
        "- Mention a finding's source language when it matters: the corpus covers 9\n"
        "  languages on purpose.\n"
        "- Plain words. No dashes as punctuation (use commas, colons or full stops).\n"
        "- NEVER use: " + ", ".join(f'"{p}"' for p in BANNED_PHRASES[:10])
    )


def _scratch_file(corpus_dir, name):
    """The text of scratchpad/<name>.md, or "" if it doesn't exist."""
    p = Path(corpus_dir) / f"{name}.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _plain_words(text, title, api_key, model):
    """
    Remove banned phrases (LLM-speak, buzzwords) from one section.
    Deleting a phrase from the middle of a sentence breaks it ("Furthermore,
    a paradigm shift looms." became ", a looms."), so:
      1. Sentence-opening transitions ("Moreover,") are removed in code.
      2. A sentence with any other banned phrase is rewritten by the AI. The
         rewrite is used only if it kept every number and [n] tag and dropped
         the phrase; otherwise the sentence stays as it was.
    """
    text, n_trans = _drop_transitions(text)
    bad = _sentences_with_banned(text)
    n_fixed = 0
    if bad:
        reply = call_ai(
            "Rewrite each sentence below without the phrase(s) in brackets. "
            "Keep its meaning, every fact, number and [n] tag. For EACH one, "
            "output exactly two lines:\nORIGINAL: <the sentence, copied exactly>\n"
            "FIX: <the rewritten sentence>\n\n"
            + "\n".join(f"{sent}   [{', '.join(ph)}]" for sent, ph in bad),
            api_key, model, temp=0.3, max_tokens=4000)
        wanted = {sent: ph for sent, ph in bad}
        for original, fix in re.findall(r"ORIGINAL:\s*(.+?)\s*\n\s*FIX:\s*(.+)", reply or ""):
            original, fix = original.strip().strip('"'), fix.strip().strip('"')
            if original in wanted and _keeps_facts(original, fix, wanted[original]):
                text = text.replace(original, fix)
                n_fixed += 1
    if n_trans or bad:
        print(f"   {title}: {n_trans} transitions removed, "
              f"{n_fixed} of {len(bad)} sentences with buzzwords rewritten")
    return text


def _drop_repeats_between(sections):
    """Drop sentences a later section repeats from an earlier one (the first stays)."""
    out, so_far = [], ""
    for title, text in sections:
        text = _drop_repeated_sentences(text, so_far)
        out.append((title, text))
        so_far += "\n\n" + text
    return out


def _summarize(question, themes, checked_text, api_key, model):
    """
    Write the bottom line and one key finding per theme from the sections the
    gate already checked. Returns (bottom line, [key finding per theme], with
    "" where the AI gave none). Themes the first answer skipped get a second,
    smaller call (a live run once stopped after theme 4 of 6).
    """
    extra = "\n\n".join(f"{t.upper()}:\n{checked_text[t]}"
                        for t in ("Where the evidence disagrees", "Gaps in the evidence")
                        if checked_text.get(t))
    key_rule = ("the key finding of that theme in 1-2 sentences: its most important specific "
                "result, with the number and who found it (\"A GitHub study found developers "
                "finished 55% faster\"), and whether other sources agree or disagree. Keep the "
                "[n] citations of the sentences it comes from, at most 3 per sentence.")

    def sections_for(numbers):
        """The checked text of the themes with these numbers, labelled THEME n."""
        return "\n\n".join(f"THEME {n}: {themes[n - 1][0]}\n{checked_text.get(themes[n - 1][0], '')}"
                           for n in numbers if checked_text.get(themes[n - 1][0]))

    all_numbers = list(range(1, len(themes) + 1))
    reply = call_ai(
        f"RESEARCH QUESTION: {question}\n\n"
        "Below are the checked sections of a research brief. Using ONLY what they say:\n"
        "BOTTOM LINE: 3-5 sentences that answer the research question: the specific results "
        "that carry the answer, how strong the evidence is, and the main tension or "
        "uncertainty.\n"
        f"K1: {key_rule}\n"
        f"K2: ... one line per theme, in order, up to K{len(themes)}.\n"
        "Plain words, no dashes as punctuation. No new facts, numbers or sources.\n\n"
        f"{sections_for(all_numbers)}\n\n{extra}",
        api_key, model, temp=0.2) or ""
    m = re.search(r"BOTTOM LINE\W*:?\s*(.+?)(?=^\W*K1\W*:|\Z)", reply, re.S | re.M | re.I)
    bottom_line = m.group(1).strip() if m else ""
    keys = dict(re.findall(r"^\W*K(\d+)\W*:?\s*(.+)$", reply, re.M | re.I))

    missing = [n for n in all_numbers if not keys.get(str(n), "").strip()
               and checked_text.get(themes[n - 1][0])]
    if missing:
        again = call_ai(
            f"RESEARCH QUESTION: {question}\n\n"
            "Below are checked sections of a research brief. For EACH theme below, write one "
            f"line starting with its label (K{missing[0]}: ...): {key_rule}\n"
            "Plain words, no dashes as punctuation. No new facts, numbers or sources.\n\n"
            f"{sections_for(missing)}",
            api_key, model, temp=0.2) or ""
        for n, text in re.findall(r"^\W*K(\d+)\W*:?\s*(.+)$", again, re.M | re.I):
            if int(n) in missing:
                keys[n] = text
    return bottom_line, [keys.get(str(n), "").strip() for n in all_numbers]


def _business_stakes(question, bottom_line, key_findings, checked_text, api_key, model):
    """
    The section 'Business stakes and our view': the 4-6 business stakes
    (what is at play for AUDIENCE: money, risk, competitive position,
    people, regulation) that the evidence raises, each with the cited
    evidence, what it means in business terms, and the authors' view.
    Written from the checked sections only, so every fact it uses has
    already passed the gate; the gate then checks this section too.
    """
    keys = "\n".join(f"{n}. {k}" for n, k in enumerate(key_findings, 1) if k)
    sections = "\n\n".join(f"{t.upper()}:\n{c}" for t, c in checked_text.items() if c)
    if not sections:
        return ""
    return call_ai(
        f"RESEARCH QUESTION: {question}\nREADERS: {AUDIENCE}\n\n"
        f"BOTTOM LINE:\n{bottom_line}\n\nKEY FINDINGS:\n{keys}\n\n"
        "Write the section 'Business stakes and our view' of this research brief, for the "
        "readers above. First decide the 4-6 business stakes this evidence raises for them: "
        "decisions or exposures where money, risk, competitive position, people or regulation "
        "is at play, for their own organisation, for the firms they invest in or lend to, or "
        "for the clients they advise. Choose only stakes the evidence speaks to, most important "
        "first.\n"
        "For each stake, write exactly:\n"
        "### <the stake as a business question, at most 12 words>\n"
        "Evidence: <2-4 sentences: the findings from the sections below that bear on it, each "
        "with its [n] citations as given, including findings that cut the other way and what "
        "stays uncertain>\n"
        "At stake: <1-3 sentences: what this means for the readers in business terms (cost, "
        "revenue, risk, competitive position, people, regulation). No new facts or numbers.>\n"
        "Our view: <1-3 sentences: a clear position on what the readers should do, expect or "
        "watch; how confident we are (high, medium or low, from how strong the evidence is); "
        "and what evidence would change the view.>\n\n"
        "A fact must come from the sections below, with its [n]; never from memory. Plain "
        "words, no dashes as punctuation.\n\n"
        f"SECTIONS OF THE BRIEF:\n{sections}",
        api_key, model, temp=0.3) or ""


def _corpus_profile(corpus, usable, by_idx):
    """The corpus counted in code: sources, languages and source types."""
    from collections import Counter
    langs = Counter(s.get("lang", "?") for s in corpus)
    kinds = Counter(_kind(s) for s in corpus)
    vendor = sum(1 for s in corpus if "vendor" in (s.get("quality") or "").lower())
    return (f"Sources: {len(corpus)} found, {len(usable) or len(corpus)} with usable content.\n"
            f"Languages: {', '.join(f'{k} {n}' for k, n in langs.most_common())}.\n"
            f"Source types: {', '.join(f'{k} {n}' for k, n in kinds.most_common())}.\n"
            f"Sources from AI tool makers: {vendor} of {len(corpus)} "
            f"({100 * vendor // max(1, len(corpus))}%).")


def _kind(source):
    """A source's type in plain words, e.g. "AI tool makers (GitHub, OpenAI...)"."""
    return source_label(source.get("quality"))


def _evidence_line(text, by_idx):
    """'Evidence: 24 sources (9 news, 6 academic, ...), in 7 languages.' for a section."""
    from collections import Counter
    cited = {n for n in verify.citations(text) if n in by_idx}
    if not cited:
        return "Evidence: no sources cited."
    kinds = Counter(_kind(by_idx[n]) for n in cited)
    langs = {by_idx[n].get("lang", "?") for n in cited}
    return (f"Evidence: {len(cited)} sources ({', '.join(f'{n} {k}' for k, n in kinds.most_common(4))}"
            f"{', ...' if len(kinds) > 4 else ''}), in {len(langs)} language"
            f"{'s' if len(langs) != 1 else ''}.")


def _method_section(topic, corpus, usable, by_idx, stats, findings, n_assigned, verdict,
                    model, corpus_dir):
    """How the brief was made. Written in code from the run's own counts, with no AI call."""
    from collections import Counter
    langs = Counter(s.get("lang", "?") for s in corpus)
    kinds = Counter(_kind(s) for s in corpus)
    n_claims = stats["ok"] + stats["corrected"] + stats["dropped"]
    tests = [line.replace("In words:", "").strip()
             for line in _scratch_file(corpus_dir, "stats_report").splitlines()
             if line.startswith("In words:")]
    return "\n\n".join([
        f"This brief was made on {datetime.now():%Y-%m-%d} by the Deep Research Toolkit, "
        f"with no person editing the text. Topic: {topic}.",
        f"**Sources.** {len(corpus)} sources were found by web, academic and forum search in "
        f"{len(langs)} languages ({', '.join(f'{k} {n}' for k, n in langs.most_common())}); "
        f"{len(usable) or len(corpus)} had usable content. Source types: "
        f"{', '.join(f'{k} {n}' for k, n in kinds.most_common())}.",
        f"**Claims.** The AI listed each usable source's factual claims with a quote from the "
        f"source: {n_claims} claims. Code checked that every quote is in its source, and a "
        f"local model checked that each quote supports its claim: {stats['ok']} were "
        f"confirmed, {stats['corrected']} corrected to match their quote, and "
        f"{stats['dropped']} dropped. Reports of the same finding were merged into "
        f"{len(findings)} findings, {sum(len(f['sources']) > 1 for f in findings)} of them "
        f"backed by two or more sources. The brief uses the {n_assigned} findings most "
        f"related to its themes.",
        f"**Writing and checking.** {model} wrote the brief from those findings only. Every "
        f"sentence then went through a fact-check gate: rules in code (each number must appear "
        f"in the cited sources, at most 3 sources per claim, no repeats), a local model that "
        f"judges meaning, and a review by {verify.CHECK_MODEL}. Result: "
        f"{verify.verdict_line(verdict)}.",
    ] + (["**Statistical tests on the corpus.** " + " ".join(tests)] if tests else []))


def _cite_from_sections(text, sections_text, most=3, need=0.6):
    """
    Give each uncited sentence of `text` the citations of the cited sentences
    in `sections_text` that mean the most nearly the same (the two best
    matches with similarity >= need, at most `most` sources). Sentences with
    no close match stay uncited. Used for the bottom line, key findings and
    business stakes, which use sections the gate already checked.
    """
    model = verify._match_model()
    if model is None or not text:
        return text
    from sentence_transformers import util
    cited = [s for s in verify.statements(sections_text) if verify.citations(s)]
    if not cited:
        return text
    plain = [re.sub(r"\[[\d,\s]+\]", "", s) for s in cited]
    emb = model.encode(plain, convert_to_tensor=True)
    out = []
    for line in text.split("\n"):
        fixed = []
        for sentence in re.split(r"(?<=[.!?])[ \t]+", line):
            if verify.citations(sentence) or len(re.findall(r"\w+", sentence)) < 8:
                fixed.append(sentence)
                continue
            sims = util.cos_sim(model.encode(sentence, convert_to_tensor=True), emb)[0]
            best = [i for i in sims.argsort(descending=True)[:2].tolist() if float(sims[i]) >= need]
            cites = list(dict.fromkeys(n for i in best for n in verify.citations(cited[i])))[:most]
            if cites:
                sentence = re.sub(r"([.!?]?)$", " [" + ", ".join(map(str, cites)) + r"]\1",
                                  sentence.rstrip(), count=1)
            fixed.append(sentence)
        out.append(" ".join(fixed))
    return "\n".join(out)


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

    # An ID placed after the full stop ("... faster. {C12}") belongs to that
    # sentence, so it is moved before the full stop. Left where it is, it would
    # become a stray "[166]" fragment of its own.
    text = re.sub(r"([.!?])((?:\s*[{\[(]\s*C\s*\d+[^}\])\n]*[}\])])+)", r"\2\1", text)
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
    that states all of them and is closest in meaning (this section's findings
    are searched first), and cite its sources instead. Returns (text, log lines).
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
            # Uncited: re-cite a sentence only if it closely matches one of this
            # section's findings (the writer left off its ID). The writer's own
            # reasoning stays uncited.
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
    itself: word for word (ignoring [n] tags, case and punctuation), or, when
    the sentence-matching model is installed, in other words with the same
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
    The brief's reference list, one APA 7th edition reference per source
    (core.apa_reference): the author(s) the page names, or the organization
    behind the site; the publication date, or "n.d." when the page gives
    none; the title, with its English translation in square brackets for a
    non-English source; the URL.

    We read corpus.json instead of sources_meta.json, so each reference number
    is the same "idx" the evidence block uses for its [n] tags. deepen
    appends new sources to corpus.json only, so numbering by the metadata
    file would leave the new [n] tags pointing at missing references.
    """

    corpus_path = os.path.join(corpus_dir, "corpus.json")

    if not os.path.exists(corpus_path):
        return ""

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)  # List of sources, each with its "idx"
    load_title_translations(corpus, corpus_dir)   # English titles of non-English sources

    # Each entry keeps the source's number, the same "idx" the [n] tags use
    entries = [f"[{s.get('idx', pos)}] {apa_reference(s)}" for pos, s in enumerate(corpus, 1)]

    # Join all entries with double newlines between them
    return "\n\n".join(entries)
