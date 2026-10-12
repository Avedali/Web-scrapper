"""
claims.py -- The claim ledger: every fact the brief may use, tied to its source
===============================================================================

Before anything is written, every usable source is read once and its factual
claims are listed, each with the exact sentence of the source that states it:

  1. EXTRACT  The AI lists each source's claims (in English) with a quote (in
              the source's language). Cached in scratchpad/claims.json, so
              re-running synthesize doesn't pay for this again.
  2. CHECK    A: the quote must really be in the source (verify.quote_in_source)
                 -- otherwise the AI made it up, and the claim is dropped.
              B: the quote must support the claim BY MEANING (the local judge).
                 A claim that overstates its quote ("the sky is blue" from "the
                 sky is often perceived as blue") is corrected by the AI to the
                 quote's hedge and scope, not deleted.
  3. MERGE    Sources that report the same finding ("up to 55% faster") become
              ONE claim backed by several sources -- so it's written once.
  4. ASSIGN   Each merged claim goes to the ONE section it fits best. A claim
              the conclusion doesn't own can't be restated in the conclusion.

The writer then cites claims, not sources: {C12}. Code turns that into the
claim's sources [3, 17] -- so a number can't end up on the wrong source.
"""

import json
import os
import re

from core import call_ai, make_batches, ai_map, _content_key, SCRATCHPAD
import verify

CLAIMS_FILE = "claims.json"
CLAIMS_PER_SOURCE = 10       # Most a source contributes (a long report has more facts)
MERGE_SIMILARITY = 0.85      # How alike two claims must be (by meaning) to be one finding
MIN_RELEVANCE = 0.20         # A claim less related than this to every section is left out
MAX_CLAIMS_PER_SECTION = 45  # Scope, not length: ~45 findings is a long section already
MAX_PER_SOURCE_IN_SECTION = 4  # Keeps one prolific source from filling a section


# ════════════════════════════════════════════════════════════════════════
# 1-2. EXTRACT AND CHECK
# ════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = (
    "Below are research sources about AI coding assistants, each starting with its "
    "number like [12]. For EACH source list its factual claims (at most MAX_N): findings, "
    "numbers, events, what a study or survey found. Skip opinions, marketing slogans "
    "and navigation text.\n"
    "Format, exactly:\n"
    "[12] CLAIM: <one sentence IN ENGLISH. Keep who says it -- the survey, study, company "
    "or author named in or around the quote -- every number, every hedge (up to, often, "
    "may) and the scope (sample size, country, company, year).>\n"
    "[12] QUOTE: <the sentence of the source that states it, copied character for "
    "character in the source's own language -- do not translate or shorten it>\n"
    "If a source has no factual claims, write: [12] NONE\n"
    "Example:\n"
    "[12] CLAIM: According to Stack Overflow's 2025 Developer Survey, 84% of developers use "
    "or plan to use AI tools, up from 76% the year before.\n"
    "[12] QUOTE: El 84% de los desarrolladores ya usa o planea usar herramientas de IA, frente "
    "al 76% del año anterior (Stack Overflow Developer Survey 2025).\n\n"
)


def _parse_claims(answer, wanted):
    """Read '[n] CLAIM: ... / [n] QUOTE: ...' pairs out of an extraction answer."""
    found, pending = {}, {}
    for line in answer.splitlines():
        m = re.match(r"^\s*\**\[(\d+)\]\**\s*(CLAIM|QUOTE|NONE)\s*:?\s*(.*)$", line.strip())
        if not m or int(m.group(1)) not in wanted:
            continue
        idx, kind, text = int(m.group(1)), m.group(2), m.group(3).strip().strip('"“”')
        found.setdefault(idx, [])
        if kind == "CLAIM":
            pending[idx] = text
        elif kind == "QUOTE" and pending.get(idx):
            found[idx].append({"claim": pending.pop(idx), "quote": text})
    return found


def extract_claims(corpus, api_key, model, corpus_dir=SCRATCHPAD):
    """
    Return {idx: [{"claim", "quote"}...]} for every usable source, using and
    updating the cache in corpus_dir/claims.json. Claims are NOT checked here.
    """
    path = os.path.join(corpus_dir, CLAIMS_FILE)
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [s for s in corpus if cache.get(str(s["idx"]), {}).get("key") != _content_key(s)]
    for attempt in range(2):   # The second pass retries sources the AI skipped
        if not todo:
            break
        print(f"   Listing the claims in {len(todo)} sources..." if attempt == 0 else
              f"   Retrying {len(todo)} sources the AI skipped...")
        parts = [(f"[{s['idx']}] ({s.get('lang', '?')}) {s.get('title', '')} -- {s.get('domain', '')}",
                  s.get("content", "")) for s in todo]
        answers = ai_map(make_batches(parts, batch_chars=60000),
                         lambda batch: EXTRACT_PROMPT.replace("MAX_N", str(CLAIMS_PER_SOURCE)) + batch,
                         api_key, model, temp=0)
        wanted = {s["idx"]: s for s in todo}
        for idx, items in _parse_claims("\n".join(answers), wanted).items():
            cache[str(idx)] = {"key": _content_key(wanted[idx]), "claims": items[:CLAIMS_PER_SOURCE],
                               "checked": False}
        todo = [s for s in todo if cache.get(str(s["idx"]), {}).get("key") != _content_key(s)]
    if todo:
        print(f"   {len(todo)} sources have no claims (the AI calls failed or skipped them)")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return cache


def check_claims(cache, by_idx, api_key, check_model=verify.CHECK_MODEL, corpus_dir=SCRATCHPAD):
    """
    Run checks A and B on every claim not checked yet; correct the ones whose
    quote doesn't clearly support them. Each claim gets a "status":
    ok / corrected / dropped: quote not in source / dropped: contradicted /
    dropped: not supported. Saves the results into the cache.
    """
    stats = {"ok": 0, "corrected": 0, "dropped": 0}
    to_judge, to_fix = [], []
    for key, entry in cache.items():
        if entry.get("checked"):
            continue
        text = by_idx.get(int(key), {}).get("content", "")
        for c in entry["claims"]:
            if not verify.quote_in_source(c["quote"], text):
                c["status"] = "dropped: quote not in source"
            else:
                to_judge.append(c)
    if to_judge:
        print(f"   Checking {len(to_judge)} claims against their quotes (local judge)...")
        verdicts = verify.judge_with_passages([(c["claim"], [c["quote"]]) for c in to_judge])
        for c, v in zip(to_judge, verdicts):
            if v == "supported":
                c["status"] = "ok"
            elif v == "contradicted":
                c["status"] = "dropped: contradicted by its quote"
            else:
                to_fix.append(c)

    # Overstated / narrower: the AI rewrites the claim to say what its quote says
    if to_fix:
        print(f"   Correcting {len(to_fix)} claims their quotes don't clearly support...")
        for start in range(0, len(to_fix), 40):
            batch = to_fix[start:start + 40]
            reply = call_ai(
                "Each CLAIM below was written from the QUOTE under it, but the quote doesn't "
                "clearly support it. For each, answer on one line:\n"
                "  K1: <the claim rewritten IN ENGLISH to say only what the quote says -- same "
                "numbers, same hedges (often, up to, may), same scope (which study, sample, "
                "company, country); keep who says it>\n"
                "  K1: DROP   (if the quote supports no factual claim)\n\n"
                + "\n\n".join(f"K{i}: CLAIM: {c['claim']}\n    QUOTE: {c['quote']}"
                              for i, c in enumerate(batch, 1)),
                api_key, check_model, temp=0, max_tokens=8000)
            answers = dict(re.findall(r"^\s*\**K(\d+)\**\s*:\s*(.+)$", reply or "", re.M))
            for i, c in enumerate(batch, 1):
                new = answers.get(str(i), "").strip()
                if not new:
                    c["status"] = "dropped: not supported (correction failed)"
                elif new.upper().startswith("DROP"):
                    c["status"] = "dropped: not supported"
                elif re.sub(r"\W+", " ", new).strip().lower() == re.sub(r"\W+", " ", c["claim"]).strip().lower():
                    c["status"] = "ok"   # The AI confirmed it as written
                else:
                    c["original_claim"], c["claim"], c["status"] = c["claim"], new, "corrected"

    for entry in cache.values():
        entry["checked"] = True
        for c in entry["claims"]:
            s = c.get("status", "")
            stats["ok" if s == "ok" else "corrected" if s == "corrected" else "dropped"] += 1
    with open(os.path.join(corpus_dir, CLAIMS_FILE), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return stats


# ════════════════════════════════════════════════════════════════════════
# 3. MERGE THE SAME FINDING FROM SEVERAL SOURCES
# ════════════════════════════════════════════════════════════════════════

QUALITY_RANK = ["academic", "government", "consulting", "news", "experiential", "vendor"]


def _rank(source):
    """Lower = stronger source (academic before vendor); unknown types last."""
    q = (source.get("quality") or "").lower()
    return next((i for i, name in enumerate(QUALITY_RANK) if name in q), len(QUALITY_RANK))


def merge_claims(cache, by_idx):
    """
    Turn the checked claims into findings: [{"id": "C1", "claim", "sources":
    [idx...] (strongest first), "quotes": {idx: quote}}]. Claims are merged
    when they mean the same (similarity >= MERGE_SIMILARITY) AND state the
    same numbers -- "55% faster" and "26% faster" stay apart however alike.
    """
    flat = []
    for key, entry in cache.items():
        for c in entry["claims"]:
            if c.get("status") in ("ok", "corrected") and int(key) in by_idx:
                flat.append({"claim": c["claim"], "quote": c["quote"], "src": int(key)})
    if not flat:
        return []

    model = verify._match_model()
    groups = []
    if model is not None:
        from sentence_transformers import util
        emb = model.encode([c["claim"] for c in flat], batch_size=64, convert_to_tensor=True)
        communities = util.community_detection(emb, threshold=MERGE_SIMILARITY, min_community_size=1)
        for community in communities:
            by_numbers = {}
            for i in community:   # Same meaning AND same numbers
                by_numbers.setdefault(frozenset(verify.numbers(flat[i]["claim"])), []).append(i)
            groups += list(by_numbers.values())
    else:
        seen = {}
        for i, c in enumerate(flat):   # Without the matching model: identical text only
            seen.setdefault(re.sub(r"\W+", " ", c["claim"].lower()), []).append(i)
        groups = list(seen.values())

    findings = []
    for members in groups:
        members = sorted(members, key=lambda i: _rank(by_idx[flat[i]["src"]]))
        sources = list(dict.fromkeys(flat[i]["src"] for i in members))
        findings.append({
            "claim": flat[members[0]]["claim"],     # Worded as the strongest source has it
            "sources": sources,
            "quotes": {flat[i]["src"]: flat[i]["quote"] for i in members},
        })
    findings.sort(key=lambda f: -len(f["sources"]))
    for n, f in enumerate(findings, 1):
        f["id"] = f"C{n}"
    return findings


# ════════════════════════════════════════════════════════════════════════
# 4. GIVE EACH FINDING TO ONE SECTION
# ════════════════════════════════════════════════════════════════════════

def outline_blocks(outline, titles):
    """{title: the outline text under that title} -- what each section is about."""
    lines, blocks, current = outline.split("\n"), {t: t for t in titles}, None
    for line in lines:
        hit = next((t for t in titles if t.lower() in line.lower()), None)
        if hit:
            current = hit
        elif current and line.strip():
            blocks[current] += "\n" + line.strip()
    return blocks


def assign_claims(findings, titles, outline):
    """
    {title: [finding...]} -- each finding in the ONE section it's most related
    to (by meaning, against the section's outline text). Findings unrelated to
    every section are left out; a section keeps its MAX_CLAIMS_PER_SECTION most
    related findings, at most MAX_PER_SOURCE_IN_SECTION led by one source.
    Returns (assignment, number of related findings left unused).
    """
    blocks = outline_blocks(outline, titles)
    model = verify._match_model()
    assignment = {t: [] for t in titles}
    if model is None or not findings:
        for i, f in enumerate(findings):   # Without the model: round-robin
            assignment[titles[i % len(titles)]].append((1.0, f))
    else:
        from sentence_transformers import util
        sims = util.cos_sim(model.encode([f["claim"] for f in findings], convert_to_tensor=True),
                            model.encode([blocks[t] for t in titles], convert_to_tensor=True))
        for i, f in enumerate(findings):
            row = sims[i].tolist()
            best = max(range(len(titles)), key=lambda k: row[k])
            if row[best] >= MIN_RELEVANCE:
                assignment[titles[best]].append((row[best], f))

    unused, out = 0, {}
    for t, items in assignment.items():
        kept, per_source = [], {}
        for score, f in sorted(items, key=lambda x: -x[0]):
            lead = f["sources"][0]
            if len(kept) < MAX_CLAIMS_PER_SECTION and per_source.get(lead, 0) < MAX_PER_SOURCE_IN_SECTION:
                kept.append(f)
                per_source[lead] = per_source.get(lead, 0) + 1
            else:
                unused += 1
        out[t] = kept
    return out, unused
