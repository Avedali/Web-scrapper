"""
claims.py: claim ledger for synthesize.py, every fact tied to a source sentence.

How it works:
Extract: The AI lists each source's claims in English with a verbatim quote (cached in claims.json)
Check: Claims whose quote is missing from the source are dropped; overstated ones are rewritten
Merge: Claims with the same meaning and numbers become one finding backed by several sources
Assign: Each finding goes to one section; the writer cites it by ID ({C12})
"""

import json
import os
import re

from core import call_ai, make_batches, ai_map, _content_key, detect_lang, SCRATCHPAD
import verify

CLAIMS_FILE = "claims.json"
CLAIMS_PER_SOURCE = 10       # Most a source contributes (a long report has more facts)
TRANSLATE_BATCH = 40         # Claims translated per AI call
MERGE_SIMILARITY = 0.85      # How alike two claims must be (by meaning) to be one finding
MIN_RELEVANCE = 0.20         # A claim less related than this to every section is left out
CLAIMS_IN_BRIEF = 260        # Findings shared by all theme sections
MAX_CLAIMS_PER_SECTION = 60  # Cap per section
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
    """Return {idx: [{"claim", "quote"}...]} for every usable source, cached in claims.json."""
    path = os.path.join(corpus_dir, CLAIMS_FILE)
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [s for s in corpus if cache.get(str(s["idx"]), {}).get("key") != _content_key(s)]
    for attempt in range(2):
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
    _translate_claims(cache, api_key, model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return cache


def _translate_claims(cache, api_key, model):
    """Translate non-English claims into English; keeps "original" and requires equal numbers."""
    def foreign(text):
        letters = re.findall(r"[^\W\d_]", text)
        latin = sum(1 for ch in letters if ch.isascii() or "\u00c0" <= ch <= "\u024f")
        return (letters and latin / len(letters) < 0.7) or detect_lang(text) not in ("en", "unknown")

    todo = [(entry, c) for entry in cache.values() for c in entry.get("claims", [])
            if "original" not in c and foreign(c["claim"])]
    if not todo:
        return
    print(f"   Translating {len(todo)} claims written in their source's language...")
    done = 0
    for start in range(0, len(todo), TRANSLATE_BATCH):
        batch = todo[start:start + TRANSLATE_BATCH]
        reply = call_ai(
            "Translate each claim below into English. Keep every number, name, hedge (up to, "
            "often, may) and the scope (which study, sample, company, country, year). One line "
            "each, starting with its label:\nT1: <translation>\n\n"
            + "\n".join(f"T{i}: {c['claim']}" for i, (_, c) in enumerate(batch, 1)),
            api_key, model, temp=0) or ""
        for m in re.finditer(r"^\W*T(\d+)\W*:\s*(.+)$", reply, re.M):
            i = int(m.group(1)) - 1
            if 0 <= i < len(batch):
                entry, c = batch[i]
                text = m.group(2).strip()
                if text and verify.numbers(text) == verify.numbers(c["claim"]):
                    c["original"], c["claim"] = c["claim"], text
                    entry["checked"] = False
                    done += 1
    print(f"   {done} of {len(todo)} translated (the rest kept their language: a number changed or the call failed)")


def check_claims(cache, by_idx, api_key, check_model=verify.CHECK_MODEL, corpus_dir=SCRATCHPAD):
    """Check unchecked claims against their quotes, set each "status" and save the cache."""
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
    if to_judge and not verify.judge_available():
        for c in to_judge:   # No local judge: quote check only
            c["status"] = "ok"
        to_judge = []
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
                    c["status"] = "ok"
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
    """Return findings [{"id", "claim", "sources", "quotes"}] merging claims of equal meaning and numbers."""
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
            # Split by numbers: a claim joins a group whose numbers include its own
            members = sorted(community, key=lambda i: -len(verify.numbers(flat[i]["claim"])))
            sub = []
            for i in members:
                nums = verify.numbers(flat[i]["claim"])
                home = next((g for g in sub if nums <= g[0]), None)
                if home:
                    home[1].append(i)
                else:
                    sub.append((nums, [i]))
            groups += [m for _, m in sub]
    else:
        seen = {}
        for i, c in enumerate(flat):
            seen.setdefault(re.sub(r"\W+", " ", c["claim"].lower()), []).append(i)
        groups = list(seen.values())

    findings = []
    for members in groups:
        most_numbers = max(len(verify.numbers(flat[i]["claim"])) for i in members)
        wording = min((i for i in members if len(verify.numbers(flat[i]["claim"])) == most_numbers),
                      key=lambda i: _rank(by_idx[flat[i]["src"]]))
        members = sorted(members, key=lambda i: _rank(by_idx[flat[i]["src"]]))
        sources = list(dict.fromkeys(flat[i]["src"] for i in members))
        findings.append({
            "claim": flat[wording]["claim"],
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
    """Return {title: outline text under that title}."""
    lines, blocks, current = outline.split("\n"), {t: t for t in titles}, None
    for line in lines:
        hit = next((t for t in titles if t.lower() in line.lower()), None)
        if hit:
            current = hit
        elif current and line.strip():
            blocks[current] += "\n" + line.strip()
    return blocks


def assign_claims(findings, titles, outline):
    """Return ({title: [finding...]}, unused count), each finding in its closest section."""
    blocks = outline_blocks(outline, titles)
    model = verify._match_model()
    assignment = {t: [] for t in titles}
    if model is None or not findings:
        for i, f in enumerate(findings):
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

    import math
    room = min(MAX_CLAIMS_PER_SECTION, math.ceil(CLAIMS_IN_BRIEF / max(1, len(titles))))
    unused, out = 0, {}
    for t, items in assignment.items():
        kept, per_source = [], {}
        for score, f in sorted(items, key=lambda x: -x[0]):
            lead = f["sources"][0]
            if len(kept) < room and per_source.get(lead, 0) < MAX_PER_SOURCE_IN_SECTION:
                kept.append(f)
                per_source[lead] = per_source.get(lead, 0) + 1
            else:
                unused += 1
        out[t] = kept
    return out, unused


# ════════════════════════════════════════════════════════════════════════
# 5. FINDINGS THAT CONTRADICT EACH OTHER
# ════════════════════════════════════════════════════════════════════════

DISAGREE_TOPIC = 0.55     # Findings at least this alike are about the same subject
DISAGREE_SURE = 0.7       # How sure the local judge must be that they contradict
MAX_DISAGREEMENTS = 8     # Pairs shown in the brief's "disagrees" section
MAX_PAIRS_JUDGED = 300    # Most alike pairs sent to the local judge


def find_disagreements(findings, most=MAX_DISAGREEMENTS):
    """Return [(finding, finding)] pairs from different sources that contradict, surest first."""
    model = verify._match_model()
    if model is None or not verify.judge_available() or len(findings) < 2:
        return []
    from sentence_transformers import util
    sims = util.cos_sim(*(2 * [model.encode([f["claim"] for f in findings], convert_to_tensor=True)]))
    candidates = []
    for i in range(len(findings)):
        for j in range(i + 1, len(findings)):
            if sims[i][j] >= DISAGREE_TOPIC and not set(findings[i]["sources"]) & set(findings[j]["sources"]):
                candidates.append((float(sims[i][j]), i, j))
    candidates = sorted(candidates, reverse=True)[:MAX_PAIRS_JUDGED]
    if not candidates:
        return []
    probs = verify.judge_support_many(
        [(findings[i]["claim"], findings[j]["claim"]) for _, i, j in candidates]
        + [(findings[j]["claim"], findings[i]["claim"]) for _, i, j in candidates])
    n = len(candidates)
    scored = sorted(((max(probs[k]["contradiction"], probs[k + n]["contradiction"]), i, j)
                     for k, (_, i, j) in enumerate(candidates)), reverse=True)
    pairs, used = [], set()
    for score, i, j in scored:
        if score < DISAGREE_SURE or len(pairs) >= most:
            break
        if i not in used and j not in used:
            pairs.append((findings[i], findings[j]))
            used.update((i, j))
    return pairs

