"""
deepen.py: Mode 1.c, targeted research to fill the brief's gaps.
=================================================================
Run: python agent.py deepen --key KEY
Or:  launcher.py → option 5

This mode reads what synthesize produced, finds the sections with weak
evidence, and searches the web again to fill them.

Example: if the brief's claim about junior developers rests on one blog
post, the AI flags it and suggests searches that could find a study.

The process:
  1. Load the research question, themes and brief from the scratchpad
  2. Ask the AI: "which sections have thin evidence?"
  3. The AI suggests specific search queries to fill each gap
  4. Run those searches and fetch new sources
  5. Add the new sources to the existing corpus
  6. Red-team the expanded corpus to check if the gaps are filled

After this, run analyze → stats → synthesize again to rebuild the brief
with the new sources.

Run it after synthesize, because it reads synthesize's output.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # Read/write structured data files
import os    # File/folder operations
import sys   # Stop with an error code when a step can't run


# From our shared toolbox:
from core import (
    search_many,      # Run several web searches at once
    region_for,       # Search region matching a query's script (ja, ko, ...)
    fetch_pages,      # Read many web pages at once
    PAGE_META,        # Author, site and date of each page read
    CONTENT_CHARS,    # How much text to keep per source
    search_academic,  # Search Semantic Scholar + arXiv
    call_ai,          # Send a prompt to the AI
    pad,              # Save to scratchpad
    classify_source,  # Label source quality (academic? vendor? etc.)
    detect_lang,      # Detect what language a text is in
    extract_search_queries,  # Pull "QUERY: ..." lines out of AI text
    QUERY_FORMAT,            # How we ask the AI to mark its search queries
    init_search,      # Set up search engines
    SCRATCHPAD,       # Path to the scratchpad folder
)


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_deepen(args):
    """
    Find evidence gaps in the current brief and fill them
    with targeted research.
    """

    # ── Setup ───────────────────────────────────────────────────────
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key found -- run install.py to add one (or pass --key KEY)")
        return

    model = args.model
    init_search(args.serper, args.brave)  # Set up search engines

    print("=" * 60)
    print("MODE 1.c: DEEPEN (fill the brief's evidence gaps)")
    print("=" * 60)

    # ── Load what synthesize produced ───────────────────────────────
    # Two files from the scratchpad:
    #   - the framing: the research question and the themes of the brief
    #   - the brief's sections, including "Gaps in the evidence"
    framing = _load_pad("synthesis_01_framing")
    brief = _load_pad("synthesis_03_brief_raw")

    if not framing:
        sys.exit("No brief found. Run 'synthesize' first.")

    # ── Load the existing corpus ────────────────────────────────────
    # We need to know what sources we already have so we:
    #   1. Don't search for things we already found
    #   2. Can add new sources to the existing collection
    #   3. Can number new sources starting after the existing ones

    corpus_path = os.path.join(SCRATCHPAD, "corpus.json")

    if os.path.exists(corpus_path):
        # Load the existing corpus
        with open(corpus_path, encoding="utf-8") as f:
            existing = json.load(f)  # List of source dictionaries

        # Build a set of the URLs already in the corpus, to check for duplicates.
        # A set differs from a list in two ways: 1) it holds no duplicates,
        # 2) checking "is X in the set?" takes the same time however big the set
        # is (O(1), against O(n) for a list). With 80 items the speed makes no
        # difference, but a set is the usual tool for this.
        #
        # {item.get("url","") for item in existing} is a "set comprehension":
        # it builds a set from the "url" of each item in existing.
        existing_urls = {item.get("url", "") for item in existing}
    else:
        existing = []      # No existing corpus
        existing_urls = set()  # set() creates an empty set

    # ════════════════════════════════════════════════════════════════
    # STEP 1: FIND THE EVIDENCE GAPS
    # ════════════════════════════════════════════════════════════════
    # We show the AI the research question, the themes and the brief, and
    # ask which themes have thin evidence and what is missing.
    #
    # The AI reads the brief and notices things like:
    #   - "Section 3 cites only one vendor study for a major claim"
    #   - "The workforce section has no data from outside the US"
    #   - "Claim [7] says '42% of companies' but source [7] doesn't
    #      contain that number"
    #
    # For each gap, it suggests specific search queries to fill it.

    print(f"\n{'─'*50}\nFINDING EVIDENCE GAPS\n{'─'*50}")

    gaps = call_ai(
        # The AI gets the framing and the whole brief. (They used to be cut to
        # 1,500-3,000 characters, which is about the first section of a long
        # brief, so gaps in later sections were never found.)
        f"You are reviewing a research brief for evidence gaps.\n\n"
        f"RESEARCH QUESTION AND THEMES:\n{framing}\n\n"
        f"BRIEF (its section 'Gaps in the evidence' lists known gaps):\n{brief}\n\n"
        f"Which themes have THIN evidence? For each weak theme:\n"
        f"1. Name the theme\n"
        f"2. What specific claim needs better support\n"
        f"3. Suggest 2-3 targeted search queries that would find that evidence\n"
        f"   (at least one in a non-English language if the gap is geographic)\n\n"
        f"Also flag: any claims tagged [n] where the source might not actually "
        f"support the claim being made.\n\n" + QUERY_FORMAT,
        api_key, model, temp=0.4,
    )

    # Save the gap analysis to the scratchpad
    pad("deepen_01_gaps", f"# Evidence Gaps\n\n{gaps}")

    # Show the first 1500 characters so the user can see what was found
    print(gaps[:1500])

    # ── Extract search queries from the AI's response ───────────────
    # The prompt asks the AI to write each query on a "QUERY: ..." line, and
    # extract_search_queries reads those lines. If the model ignored the
    # format, it falls back to quoted strings such as 'search for "junior
    # developer AI tool adoption rates 2025"'.
    fix_queries = extract_search_queries(gaps)

    print(f"\n   Found {len(fix_queries)} targeted queries")

    # ════════════════════════════════════════════════════════════════
    # STEP 2: RUN TARGETED SEARCHES
    # ════════════════════════════════════════════════════════════════
    # For each query the gap analysis suggested, search the web and
    # academic databases. Only keep results we don't already have.
    #
    # We limit to 12 queries to keep runtime reasonable.
    # fix_queries[:12] = the first 12 queries (Python slice syntax).

    print(f"\n{'─'*50}\nTARGETED SEARCH ({len(fix_queries[:12])} queries)\n{'─'*50}")

    new_sources = []  # Holds the sources that are not already in the corpus
    queries = fix_queries[:12]

    def add(item):
        """Number the new source after the existing ones and keep it."""
        item["idx"] = len(existing) + len(new_sources) + 1
        new_sources.append(item)
        existing_urls.add(item["url"])

    # ── Web search: all queries at once, then read new pages in parallel
    results = search_many([(q, 3, region_for(q)) for q in queries])
    found = {}  # url -> search result (first query that found it)
    for q, rs in zip(queries, results, strict=True):
        print(f"   {q[:55]}...  ({len(rs)} results)")
        for r in rs:
            if r["url"] not in existing_urls and r["url"] not in found:
                found[r["url"]] = r
    pages = fetch_pages(list(found))
    for url, r in found.items():
        if pages.get(url):
            add({
                "title": r["title"],
                "domain": r["domain"],
                "url": url,
                "content": pages[url][:CONTENT_CHARS],     # Cut long pages to CONTENT_CHARS
                "quality": classify_source(r["domain"]),   # academic? vendor? etc.
                "lang": detect_lang(pages[url]),           # What language?
                **PAGE_META.get(url, {}),                  # Author(s), site name, date (for APA)
            })
            print(f"      {r['domain']} [{new_sources[-1]['lang']}]")

    # ── Academic search (for research-type queries only) ────────────
    # If the query contains words like "study", "research", "evidence",
    # or "survey", it's probably looking for academic sources.
    # In that case, also search Semantic Scholar and arXiv.
    for q in queries:
        if any(kw in q.lower() for kw in ["study", "research", "evidence", "survey"]):
            for p in search_academic(q, n=2):
                if p["url"] not in existing_urls:
                    add({
                        "title": p["title"],
                        "domain": p["domain"],
                        "url": p["url"],
                        "content": p["snippet"][:CONTENT_CHARS],  # For papers, content = abstract
                        "quality": "academic",
                        "year": p.get("year"),           # Publication year (for APA)
                        "authors": p.get("authors") or [],   # Paper authors (for APA)
                        "lang": "en",
                    })
                    print(f"      {p['title'][:50]}...")

    # ════════════════════════════════════════════════════════════════
    # STEP 3: UPDATE THE CORPUS
    # ════════════════════════════════════════════════════════════════
    # Merge the new sources with the existing corpus and save.
    # The "+" operator concatenates (joins) two lists:
    # [1,2,3] + [4,5] → [1,2,3,4,5]

    full_corpus = existing + new_sources

    # Overwrite the corpus file with the expanded version
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(full_corpus, f, indent=2, ensure_ascii=False)

    # ════════════════════════════════════════════════════════════════
    # STEP 4: RED TEAM THE EXPANDED CORPUS
    # ════════════════════════════════════════════════════════════════
    # Ask the AI whether the new sources fill the gaps and which claims
    # are still weak.
    #
    # This repeats the check done by the depth-warden in the .md agents,
    # which asks whether each claim is covered until it is.

    print(f"\n{'─'*50}\nRED TEAM (post-deepen)\n{'─'*50}")

    critique = call_ai(
        f"Red-team pass on the expanded corpus ({len(full_corpus)} sources now).\n"
        f"Focus on: do the new sources actually fill the gaps identified?\n"
        f"Are there still weak claims?\n\n"
        f"New sources added:\n"
        # List the new sources so the AI can evaluate them
        + "\n".join(
            f"- [{s['quality']}][{s['lang']}] {s['title']}"
            for s in new_sources[:15]  # Show up to 15 new sources
        ),
        api_key, model, temp=0.4,
    )

    # Save the red team critique
    pad("deepen_02_red_team", f"# Post-Deepen Red Team\n\n{critique}")

    # ── Done ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DEEPEN COMPLETE")
    print(f"   Added {len(new_sources)} new sources")
    print(f"   Total corpus: {len(full_corpus)} sources")
    print("   Run 'analyze' again, then 'synthesize' to rebuild the brief.")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION
# ════════════════════════════════════════════════════════════════════════

def _load_pad(name):
    """
    Load a scratchpad file by name.

    This is a shorter version of load_pad from core.py. It does the same
    thing. deepen.py defines its own copy because it needs only this one
    helper, and a local copy keeps the file self-contained.

    Returns the file contents as a string, or "" if the file doesn't exist.
    """
    # Build the full path: e.g., "scratchpad/synthesis_01_framing.md"
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()

    return ""  # File not found → return empty string
