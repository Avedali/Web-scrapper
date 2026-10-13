"""deepen.py: Mode 1.c, targeted research to fill the brief's evidence gaps.
Run after synthesize: python agent.py deepen --key KEY (or launcher.py, option 5)
Steps: AI gap analysis, targeted web and academic search, corpus.json update, red team.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json
import os
import sys


from core import (
    search_many,
    region_for,
    fetch_pages,
    PAGE_META,
    CONTENT_CHARS,
    search_academic,
    call_ai,
    pad,
    classify_source,
    detect_lang,
    extract_search_queries,
    QUERY_FORMAT,
    init_search,
    SCRATCHPAD,
)


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_deepen(args):
    """Find evidence gaps in the current brief and fill them with targeted research."""

    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key found -- run install.py to add one (or pass --key KEY)")
        return

    model = args.model
    init_search(args.serper, args.brave)

    print("=" * 60)
    print("MODE 1.c: DEEPEN (fill the brief's evidence gaps)")
    print("=" * 60)

    # ── Load the synthesize output and the existing corpus ──────────
    framing = _load_pad("synthesis_01_framing")
    brief = _load_pad("synthesis_03_brief_raw")

    if not framing:
        sys.exit("No brief found. Run 'synthesize' first.")

    corpus_path = os.path.join(SCRATCHPAD, "corpus.json")

    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            existing = json.load(f)

        existing_urls = {item.get("url", "") for item in existing}
    else:
        existing = []
        existing_urls = set()

    # ════ STEP 1: FIND THE EVIDENCE GAPS ══════════════════════════════

    print(f"\n{'─'*50}\nFINDING EVIDENCE GAPS\n{'─'*50}")

    gaps = call_ai(
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

    pad("deepen_01_gaps", f"# Evidence Gaps\n\n{gaps}")

    print(gaps[:1500])

    fix_queries = extract_search_queries(gaps)

    print(f"\n   Found {len(fix_queries)} targeted queries")

    # ════ STEP 2: RUN TARGETED SEARCHES ═══════════════════════════════

    print(f"\n{'─'*50}\nTARGETED SEARCH ({len(fix_queries[:12])} queries)\n{'─'*50}")

    new_sources = []
    queries = fix_queries[:12]

    def add(item):
        """Number the new source after the existing ones and keep it."""
        item["idx"] = len(existing) + len(new_sources) + 1
        new_sources.append(item)
        existing_urls.add(item["url"])

    # ── Web search: all queries at once, then fetch new pages in parallel
    results = search_many([(q, 3, region_for(q)) for q in queries])
    found = {}
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
                "content": pages[url][:CONTENT_CHARS],
                "quality": classify_source(r["domain"]),
                "lang": detect_lang(pages[url]),
                **PAGE_META.get(url, {}),
            })
            print(f"      {r['domain']} [{new_sources[-1]['lang']}]")

    # ── Academic search for study/research/evidence/survey queries ──
    for q in queries:
        if any(kw in q.lower() for kw in ["study", "research", "evidence", "survey"]):
            for p in search_academic(q, n=2):
                if p["url"] not in existing_urls:
                    add({
                        "title": p["title"],
                        "domain": p["domain"],
                        "url": p["url"],
                        "content": p["snippet"][:CONTENT_CHARS],
                        "quality": "academic",
                        "year": p.get("year"),
                        "authors": p.get("authors") or [],
                        "lang": "en",
                    })
                    print(f"      {p['title'][:50]}...")

    # ════ STEP 3: UPDATE THE CORPUS ═══════════════════════════════════

    full_corpus = existing + new_sources

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(full_corpus, f, indent=2, ensure_ascii=False)

    # ════ STEP 4: RED TEAM THE EXPANDED CORPUS ════════════════════════

    print(f"\n{'─'*50}\nRED TEAM (post-deepen)\n{'─'*50}")

    critique = call_ai(
        f"Red-team pass on the expanded corpus ({len(full_corpus)} sources now).\n"
        f"Focus on: do the new sources actually fill the gaps identified?\n"
        f"Are there still weak claims?\n\n"
        f"New sources added:\n"
        + "\n".join(
            f"- [{s['quality']}][{s['lang']}] {s['title']}"
            for s in new_sources[:15]
        ),
        api_key, model, temp=0.4,
    )

    pad("deepen_02_red_team", f"# Post-Deepen Red Team\n\n{critique}")

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
    """Return scratchpad/<name>.md as a string, or "" if it does not exist."""
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()

    return ""
