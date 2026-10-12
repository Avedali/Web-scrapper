"""
verify.py -- Checks that keep the brief tied to its sources
============================================================

Two checks the claim ledger and the fact-check gate are built on:

  A. quote_in_source(quote, source)
     Is this quote really in the source? Compared word for word, but
     forgiving about formatting (spacing, curly quotes, capitals, words split
     across lines). The only thing it catches is a quote the AI made up.

  B. judge_support(premise, hypothesis)
     Does the premise (a quote from a source) support the hypothesis (a
     claim or a sentence of the brief)? Judged by MEANING, by a local model
     trained for exactly this ("natural language inference"), in 100+
     languages. It runs on this computer -- free, and independent of the AI
     that wrote the text, so it doesn't share that AI's blind spots.
       "the sky is often perceived as blue"  →  "the sky is blue"
       is NOT supported (the claim drops the hedge): the claim gets corrected
       to the quote's wording, not deleted.
"""

import re
import unicodedata
from difflib import SequenceMatcher


# ════════════════════════════════════════════════════════════════════════
# A. IS THE QUOTE REALLY IN THE SOURCE?
# ════════════════════════════════════════════════════════════════════════

QUOTE_MIN_CHARS = 15     # Shorter "quotes" ("AI tools") prove nothing
QUOTE_MATCH = 0.9        # Share of the quote that must match, in order


def normalize(text):
    """
    Text reduced to what matters for matching: same letters and numbers,
    no formatting. Curly quotes, dashes, capitals, extra spaces, soft hyphens
    and words split across lines ("produc-\\ntivity") all disappear.
    """
    text = unicodedata.normalize("NFKC", text or "")   # Full-width → normal, "ﬁ" → "fi"
    text = text.replace("­", "")                  # Soft hyphens
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # Words split across lines
    text = re.sub(r"[^\w%$€£¥.,]+", " ", text.lower())  # Punctuation → space (keep 4.5%, $1,000)
    text = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", text)  # ...but only inside numbers
    return re.sub(r"\s+", " ", text).strip()


def quote_in_source(quote, source):
    """
    True if `quote` appears in `source` -- exactly after normalizing, or with
    at least QUOTE_MATCH of its characters matching in order (pages extracted
    from HTML lose and gain the odd character).
    """
    q, s = normalize(quote), normalize(source)
    if len(q) < QUOTE_MIN_CHARS:
        return False
    if q in s:
        return True
    # Fuzzy match, made fast: find where pieces of the quote occur in the
    # source, and compare the quote with the source text around each place
    # (character pieces, so it works in Japanese and Chinese too)
    starts = set()
    for i in range(0, max(1, len(q) - 20), 10):
        piece = q[i:i + 20]
        pos = s.find(piece)
        while pos != -1 and len(starts) < 50:
            starts.add(max(0, pos - i))
            pos = s.find(piece, pos + 1)
    for start in starts:
        window = s[max(0, start - 20): start + len(q) + 20]
        matched = sum(b.size for b in SequenceMatcher(None, window, q, autojunk=False)
                      .get_matching_blocks())
        if matched >= QUOTE_MATCH * len(q):
            return True
    return False


# ════════════════════════════════════════════════════════════════════════
# B. DOES THE QUOTE SUPPORT THE CLAIM? (by meaning)
# ════════════════════════════════════════════════════════════════════════

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"   # Same model analyze uses (already downloaded)
SUPPORTED = 0.5        # Probability of "entailment" needed to count as supported
CONTRADICTED = 0.5     # Probability of "contradiction" that counts as contradicted

_nli = {}   # The loaded model, kept so it's loaded once


def _load_nli():
    """Load the model once; on an Apple Silicon Mac, on its GPU if that works."""
    if not _nli:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        device = "cpu"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            try:
                model.to("mps")
                device = "mps"
            except Exception:
                model.to("cpu")
        labels = {i: name.lower() for i, name in model.config.id2label.items()}
        _nli.update(tokenizer=tokenizer, model=model, device=device, labels=labels, torch=torch)
    return _nli


def judge_support_many(pairs, batch_size=8):
    """
    For each (premise, hypothesis) pair, the model's probabilities as
    {"entailment": p, "neutral": p, "contradiction": p}.
    """
    nli = _load_nli()
    torch, results = nli["torch"], []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        enc = nli["tokenizer"]([p for p, _ in batch], [h for _, h in batch],
                               truncation="only_first", max_length=512,
                               padding=True, return_tensors="pt")
        try:
            with torch.no_grad():
                logits = nli["model"](**{k: v.to(nli["device"]) for k, v in enc.items()}).logits
        except Exception:
            if nli["device"] == "cpu":
                raise
            # The Mac GPU failed mid-run: move to the CPU for good
            nli["model"].to("cpu")
            nli["device"] = "cpu"
            with torch.no_grad():
                logits = nli["model"](**enc).logits
        for row in logits.softmax(-1).cpu().tolist():
            results.append({nli["labels"][j]: p for j, p in enumerate(row)})
    return results


def verdict(probs):
    """'supported', 'contradicted' or 'not shown' from judge_support_many's output."""
    if probs["entailment"] >= SUPPORTED and probs["entailment"] > probs["contradiction"]:
        return "supported"
    if probs["contradiction"] >= CONTRADICTED:
        return "contradicted"
    return "not shown"   # Overstated, narrower, or about something else


def judge_support(premise, hypothesis):
    """'supported', 'contradicted' or 'not shown' for one pair."""
    return verdict(judge_support_many([(premise, hypothesis)])[0])


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYER 1-2: RULES IN CODE + THE LOCAL JUDGE
# ════════════════════════════════════════════════════════════════════════
# Works on any brief whose claims cite sources as [n] -- one synthesize just
# wrote, or one a teammate edited by hand (python agent.py verify).

MAX_CITES = 3            # A claim needs 1-3 sources; more says nothing about which
PASSAGES = 5             # Source sentences shown to the judge per claim
MATCH_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Same as BERTopic


def sentences(text):
    """Split text into sentences (keeps lists and headings as their own lines)."""
    out = []
    for line in text.split("\n"):
        out += [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", line) if x.strip()]
    return out


def citations(sentence):
    """The [n] source numbers a sentence cites."""
    return [int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence)
            for n in re.split(r"\s*,\s*", group)]


def _readings(raw):
    """
    The ways one written number can be read, in one standard form: "1,000"
    → {"1000"}; "0,47" → {"0.47"}; "1.500" is 1.5 in English but 1500 in
    German → {"1.5", "1500"}.
    """
    n = raw.replace("\u00a0", " ")
    out = set()
    if re.fullmatch(r"\d{1,3}(?:[, .]\d{3})+", n) and not n.startswith("0"):
        out.add(re.sub(r"[, .]", "", n))                 # Thousands separators
    if re.fullmatch(r"\d+[.,]\d+", n) and not (out and "," in n):
        out.add(n.replace(",", "."))                     # A decimal point or comma
        # ("1,000" is read as a thousand only; "1.500" stays ambiguous)
    if not out:
        out.add(re.sub(r"[ ,]", "", n) if re.fullmatch(r"[\d ,]+", n) else n.replace(",", "."))
    return {x.rstrip("0").rstrip(".") if "." in x else x for x in out}


def number_readings(text):
    """
    The numbers in a text, each as the set of its possible readings. Single
    digits are left out ("3 risks" is too common to check); [n] citations are
    ignored.
    """
    text = unicodedata.normalize("NFKC", re.sub(r"\[[\d,\s]+\]", " ", text or ""))
    found = {}
    for raw in re.findall(r"\d+(?:[.,\u00a0 ]\d+)*", text):
        readings = _readings(raw)
        if max(len(x.replace(".", "")) for x in readings) >= 2:
            found[raw.strip()] = readings
    return found


def numbers(text):
    """Every reading of every number in a text, as one set (for lookups)."""
    return set().union(*number_readings(text).values()) if text else set()


_matcher = {}


def _match_model():
    """The sentence-matching model (None if sentence-transformers isn't installed)."""
    if "model" not in _matcher:
        try:
            from sentence_transformers import SentenceTransformer
            _matcher["model"] = SentenceTransformer(MATCH_MODEL)
        except Exception:
            _matcher["model"] = None
    return _matcher["model"]


def best_passages(claim, source_texts, k=PASSAGES):
    """
    The k sentences of the cited sources that best match `claim` -- by
    meaning when the matching model is installed (so an English sentence finds
    its passage in a Japanese source), else by shared words and numbers.
    """
    candidates = []
    for text in source_texts:
        candidates += [s for s in sentences(text) if 25 <= len(s) <= 600]
    if not candidates:
        return []
    model = _match_model()
    if model is not None:
        from sentence_transformers import util
        cache = _matcher.setdefault("cache", {})
        todo = [c for c in candidates if c not in cache]
        if todo:
            for c, e in zip(todo, model.encode(todo, batch_size=64, convert_to_tensor=True)):
                cache[c] = e
        import torch
        scores = util.cos_sim(model.encode(claim, convert_to_tensor=True),
                              torch.stack([cache[c] for c in candidates]))[0].tolist()
    else:
        words = set(re.findall(r"\w{4,}", claim.lower())) | numbers(claim)
        scores = [len(words & (set(re.findall(r"\w{4,}", c.lower())) | numbers(c)))
                  for c in candidates]
    top = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return list(dict.fromkeys(candidates[i] for i in top))[:k]   # Best match first


def check_sentences(text, by_idx, findings="", use_judge=True):
    """
    Check every sentence of `text` (one section, or a whole brief). Returns one
    report per sentence that states something: {"sentence", "cites", "problems"
    (list of strings), "passages" (what the judge read)}.

    Rules (code): cites at most MAX_CITES sources, all of which exist; every
    number appears in the cited sources (or, for an uncited sentence -- this
    study's own results -- in the research findings); not a repeat.
    Judge (local model): the cited passages support the sentence's meaning.
    """
    findings_numbers = numbers(findings)
    seen, reports, to_judge = set(), [], []
    for s in sentences(text):
        if s.startswith(("#", "**", "---", "|")) or len(re.findall(r"\w+", s)) < 4:
            continue   # Headings, the header block, tables, stray fragments
        cites = citations(s)
        problems = []
        if len(cites) > MAX_CITES:
            problems.append(f"cites {len(cites)} sources (max {MAX_CITES})")
        missing_src = [n for n in cites if n not in by_idx]
        if missing_src:
            problems.append(f"cites sources that don't exist: {missing_src}")
        source_texts = [by_idx[n].get("content", "") for n in cites if n in by_idx]
        nums = number_readings(s)
        if nums:
            pool = set().union(*(numbers(t) for t in source_texts)) if cites else findings_numbers
            absent = sorted(raw for raw, readings in nums.items() if not readings & pool)
            if absent:
                where = "the cited sources" if cites else "the research findings (no source cited)"
                problems.append(f"numbers not in {where}: {', '.join(absent)}")
        key = re.sub(r"\W+", " ", re.sub(r"\[[\d,\s]+\]", "", s)).strip().lower()
        if len(key) > 40:
            if key in seen:
                problems.append("repeats an earlier sentence")
            seen.add(key)
        report = {"sentence": s, "cites": cites, "problems": problems, "passages": []}
        reports.append(report)
        if use_judge and source_texts and not missing_src:
            # "A [19] and B [166]" is two claims: each part is judged against
            # its own sources' passages
            for part, part_cites in claim_parts(s):
                texts = [by_idx[n].get("content", "") for n in part_cites if n in by_idx]
                passages = best_passages(part, texts)
                if passages:
                    report["passages"] += passages
                    to_judge.append((report, part, passages))

    # The local judge, all at once (batched): do the passages support each part?
    if to_judge:
        verdicts = judge_with_passages([(part, passages) for _, part, passages in to_judge])
        order = {"supported": 0, "not shown": 1, "contradicted": 2}
        for (report, _, _), v in zip(to_judge, verdicts):
            if order[v] >= order[report.get("judge", "supported")]:
                report["judge"] = v      # The sentence gets its weakest part's verdict
        for report in reports:
            if report.get("judge", "supported") != "supported":
                report["problems"].append(f"local judge: {report['judge']} by the cited passages")
    return reports


def judge_with_passages(items):
    """
    For each (claim, passages) -- passages best match first -- 'supported',
    'contradicted' or 'not shown'. Each passage is judged on its own and the
    best two together, and the strongest support wins: glued together, one
    supporting sentence got drowned out by unrelated ones. A contradiction
    only counts from the best-matching passage.
    """
    pairs, spans = [], []
    for claim, passages in items:
        options = list(dict.fromkeys(passages))           # Without duplicates
        if len(options) > 1:
            options.append(" ".join(options[:2]))
        claims = list(dict.fromkeys([claim, core_claim(claim)]))
        spans.append((len(pairs), len(options), len(claims)))
        pairs += [(premise, c) for c in claims for premise in options]
    probs = judge_support_many(pairs) if pairs else []
    out = []
    for start, n_options, n_claims in spans:
        mine = probs[start:start + n_options * n_claims]
        if max(p["entailment"] for p in mine) >= SUPPORTED:
            out.append("supported")
        elif mine[0]["contradiction"] >= CONTRADICTED:   # Best passage vs the full claim
            out.append("contradicted")
        else:
            out.append("not shown")
    return out


_LEAD_INS = [
    r"(?:and|but|while|whereas|with|however|moreover|also|similarly|meanwhile|yet)\b[,\s]*",
    r"(?:(?:one|another|a|an|some|other|recent|several|further|separate|this|that|the)\s+)?"
    r"(?:report|study|studies|analysis|analyses|survey|surveys|research|data|test|tests|"
    r"trial|experiment|paper|source|sources)\s+(?:also\s+)?(?:found|finds|indicated|indicates|"
    r"show|shows|showed|suggest|suggests|suggested|reveal|reveals|revealed|reported|reports|"
    r"noted|notes|estimates|estimated)\s+(?:that\s+)?",
    r"(?:according to [^,]+,|teams report(?:ing)?|developers report(?:ing)?)\s*",
]


def core_claim(claim):
    """
    The fact in a claim, without its lead-in: "and another study indicated
    that 45% of examples failed" → "45% of examples failed". The local judge
    trips over such lead-ins even when the passage says exactly the fact.
    """
    text = claim.split("; ")[-1].strip()
    for _ in range(3):
        before = text
        for pattern in _LEAD_INS:
            text = re.sub(r"^" + pattern, "", text, flags=re.IGNORECASE).strip()
        if text == before:
            break
    return text[:1].upper() + text[1:] if len(re.findall(r"\w+", text)) >= 3 else claim


def claim_parts(sentence):
    """
    Split a sentence at its citations into (text, [cited sources]) parts:
    "A grew 40% [19], and B fell [166, 192]." → [("A grew 40%", [19]),
    ("and B fell", [166, 192])]. Text after the last citation is ignored.
    """
    parts, start = [], 0
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence):
        text = sentence[start:m.start()].strip(" ,;:")
        cites = [int(n) for n in re.split(r"\s*,\s*", m.group(1))]
        if len(re.findall(r"\w+", text)) >= 3:
            parts.append((text, cites))
        elif parts:
            parts[-1] = (parts[-1][0], parts[-1][1] + cites)   # "[3][4]" → one part
        start = m.end()
    return parts


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYER 3: AI REVIEW AND REPAIR
# ════════════════════════════════════════════════════════════════════════
# The local judge flags; an AI decides, with the passages in front of it.
# A stronger model than the writer: in a test, the default model deleted
# 25 of 25 flagged sentences, this one corrected 19 of 24.

CHECK_MODEL = "google/gemini-2.5-flash"

REVIEW_RULES = (
    "Rules:\n"
    "- A sentence is SUPPORTED only if its passages say the same thing, with the same "
    "numbers, the same hedges ('often', 'up to', 'may') and the same scope: one "
    "company's or one study's sample presented as 'developers' or 'companies' in "
    "general is NOT supported.\n"
    "- A vendor's own claim must be attributed to it ('GitHub reports ...').\n"
    "- Prefer FIX to DELETE: correct the number, word, hedge or scope, or keep only "
    "the part the passages support. Use only the passages given; never add a claim. "
    "Keep [n] citations, at most 3 per sentence, only to sources whose passages say it.\n"
)


def _review_prompt(items, findings):
    """The AI review prompt for one section's sentences."""
    blocks = []
    for i, it in enumerate(items, 1):
        if it["cites"]:
            passages = "\n".join(f"    · {p}" for p in it["passages"]) or "    (no passage found)"
            blocks.append(f"S{i}: {it['sentence']}\n  problems found: {'; '.join(it['problems'])}\n"
                          f"  passages from its cited sources:\n{passages}")
        else:
            blocks.append(f"S{i}: {it['sentence']}\n  (cites no source)")
    return (
        "You are the fact-check gate of a research brief. For each sentence below, "
        "answer on ONE line, starting with its label:\n"
        "  S1: KEEP              -- cited and supported by its passages; or uncited ARGUMENT: a "
        "topic sentence, a transition, the thesis, an explanation, or a conclusion drawn from "
        "the cited facts (these carry the argument -- keep them); or this study's own results "
        "reported correctly (see RESEARCH FINDINGS)\n"
        "  S1: FIX: <sentence>   -- corrected so its passages (or the findings) support it\n"
        "  S1: DELETE            -- a factual claim (a number, an event, what a study found) "
        "that nothing given supports\n"
        "  S1: FILLER            -- ONLY an empty phrase, or a sentence that repeats what the "
        "same paragraph already said; when unsure, KEEP\n\n"
        + REVIEW_RULES + "\n"
        f"<research_findings>\n{findings}\n</research_findings>\n\n"
        "SENTENCES:\n\n" + "\n\n".join(blocks)
    )


def review_section(text, by_idx, findings, api_key, model=CHECK_MODEL, use_judge=True):
    """
    The gate for one section: rules and the local judge flag sentences, the
    AI reviews the flagged ones and every uncited one, fixes are re-checked by
    the rules, and anything that still fails is removed. Returns
    (new text, outcomes) -- one outcome per reviewed sentence, for the log --
    or (text, None) if the AI call failed (the section is then NOT CHECKED).
    """
    from core import call_ai

    reports = check_sentences(text, by_idx, findings, use_judge=use_judge)
    # Repeats need no AI: the later copy goes
    outcomes = []
    for r in reports:
        if "repeats an earlier sentence" in r["problems"]:
            at = text.rfind(r["sentence"])          # The LATER copy goes
            if at != -1:
                text = text[:at] + text[at + len(r["sentence"]):]
            outcomes.append(f"REPEAT: {r['sentence']}\n  → DELETED")
    items = [r for r in reports if "repeats an earlier sentence" not in r["problems"]
             and (r["problems"] or not r["cites"])]
    if not items:
        return text, outcomes

    reply = call_ai(_review_prompt(items, findings), api_key, model, temp=0, max_tokens=12000)
    if not reply:
        return text, None
    answers = {}
    for m in re.finditer(r"^\s*\**S(\d+)\**\s*:\s*(KEEP|DELETE|FILLER|FIX\s*:\s*(.+))\s*$", reply, re.M):
        answers[int(m.group(1))] = (m.group(2).split(":")[0].strip().upper(), (m.group(3) or "").strip())

    for i, it in enumerate(items, 1):
        action, fix = answers.get(i, ("NO ANSWER", ""))
        hard = [p for p in it["problems"] if not p.startswith("local judge")]
        label = "; ".join(it["problems"]) or "uncited"
        if action == "KEEP" and not hard:
            if it["problems"]:
                outcomes.append(f"KEPT ({label}): {it['sentence']}\n  → the AI judged it supported")
            continue
        if action == "FIX" and fix:
            # A fix must pass the rules itself (numbers in its sources, real
            # citations, at most MAX_CITES), and the judge mustn't contradict it
            check = check_sentences(fix, by_idx, findings, use_judge=use_judge)
            bad = [p for r in check for p in r["problems"]
                   if not p.startswith("local judge") or "contradicted" in p]
            if not bad:
                text = text.replace(it["sentence"], fix, 1)
                outcomes.append(f"CORRECTED ({label}): {it['sentence']}\n  → {fix}")
                continue
            outcomes.append(f"FIX REJECTED ({'; '.join(bad)}): {it['sentence']}\n  → {fix}\n  → DELETED")
        elif action == "KEEP":
            outcomes.append(f"KEPT BY AI BUT BREAKS A RULE ({'; '.join(hard)}): {it['sentence']}\n  → DELETED")
        elif not it["cites"] and action == "NO ANSWER":
            continue   # An uncited sentence the AI didn't mention: leave it
        else:
            outcomes.append(f"{action} ({label}): {it['sentence']}\n  → DELETED")
        text = text.replace(it["sentence"], "", 1)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), outcomes


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYER 4: INDEPENDENT AUDIT AND THE VERDICT
# ════════════════════════════════════════════════════════════════════════

AUDIT_SAMPLE = 20        # Final sentences re-checked against their sources' full text
PASS_UNSUPPORTED = 0.10  # At most this share of the sample unsupported
PASS_DELETED = 0.40      # At most this share of reviewed sentences deleted


def audit(text, by_idx, api_key, model=CHECK_MODEL, n=AUDIT_SAMPLE, seed=0):
    """
    Re-check a random sample of the final cited sentences against the FULL
    text of their sources (not the passages the gate picked) -- the automated
    version of a human audit. Returns a list of (sentence, verdict, reason)
    with verdict SUPPORTED / PARTLY / UNSUPPORTED / ERROR.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor
    from core import call_ai

    cited = [s for s in sentences(text) if citations(s) and not s.startswith("#")]
    sample = random.Random(seed).sample(cited, min(n, len(cited)))

    def one(sentence):
        sources = "\n\n".join(
            f"[{c}] {by_idx[c].get('title', '')}\n{by_idx[c].get('content', '')}"
            for c in citations(sentence) if c in by_idx)
        reply = call_ai(
            f"<sources>\n{sources}\n</sources>\n\n<sentence>\n{sentence}\n</sentence>\n\n"
            "Does the sentence say only what its cited sources say -- same numbers, hedges "
            "and scope (one study's sample is not 'developers' in general)? Answer on one "
            "line: SUPPORTED, PARTLY or UNSUPPORTED, then ' -- ' and a reason of at most 20 words.",
            api_key, model, temp=0, max_tokens=200)
        m = re.match(r"\W*(UNSUPPORTED|PARTLY|SUPPORTED)\b\W*(.*)", (reply or "").strip(), re.S)
        return (sentence, m.group(1), m.group(2).strip()[:200]) if m else (sentence, "ERROR", reply[:100])

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, sample))


def make_verdict(audit_results, n_reviewed, n_deleted, not_checked):
    """
    PASS or NEEDS REVIEW, with the reasons and the numbers behind them.
    NEEDS REVIEW: a section wasn't checked, the audit found more than
    PASS_UNSUPPORTED unsupported, or the gate deleted more than PASS_DELETED
    of what it reviewed (a brief gutted by its own checker is badly written).
    """
    graded = [v for _, v, _ in audit_results if v != "ERROR"]
    counts = {k: graded.count(k) for k in ("SUPPORTED", "PARTLY", "UNSUPPORTED")}
    reasons = []
    if not_checked:
        reasons.append(f"{len(not_checked)} section(s) not fact-checked: {', '.join(not_checked)}")
    if not graded:
        reasons.append("the independent audit could not run")
    elif counts["UNSUPPORTED"] / len(graded) > PASS_UNSUPPORTED:
        reasons.append(f"audit: {counts['UNSUPPORTED']} of {len(graded)} sampled sentences unsupported")
    if n_reviewed and n_deleted / n_reviewed > PASS_DELETED:
        reasons.append(f"the gate deleted {n_deleted} of {n_reviewed} sentences it reviewed")
    return {
        "status": "NEEDS REVIEW" if reasons else "PASS",
        "reasons": reasons,
        "audit": counts, "audit_sample": len(graded),
        "reviewed": n_reviewed, "deleted": n_deleted,
    }


def verdict_line(v):
    """One line for the brief's header and the console."""
    a, n = v["audit"], v["audit_sample"]
    audit_text = (f"independent audit of {n} sentences: {a['SUPPORTED']} supported, "
                  f"{a['PARTLY']} partly, {a['UNSUPPORTED']} unsupported") if n else "no audit"
    return (f"{v['status']} -- {audit_text}; gate reviewed {v['reviewed']} sentences, "
            f"deleted {v['deleted']}" + (f". Why: {'; '.join(v['reasons'])}" if v["reasons"] else ""))


# ════════════════════════════════════════════════════════════════════════
# THE WHOLE GATE, AND "python agent.py verify"
# ════════════════════════════════════════════════════════════════════════

def gate(sections, by_idx, findings, api_key, model=CHECK_MODEL):
    """
    Run the gate over a brief's [(title, text)] sections. Returns
    (checked sections, log lines, verdict dict).
    """
    checked, log, not_checked, n_reviewed, n_deleted = [], [], [], 0, 0
    for title, text in sections:
        new, outcomes = review_section(text, by_idx, findings, api_key, model)
        if outcomes is None:
            not_checked.append(title)
            print(f"   {title}: NOT CHECKED -- the AI call failed (see the error above)")
            log.append(f"## {title}\n\n- NOT CHECKED: the AI review call failed")
            checked.append((title, text))
            continue
        deleted = sum(1 for o in outcomes if o.endswith("→ DELETED"))
        fixed = sum(1 for o in outcomes if o.startswith("CORRECTED"))
        n_reviewed += len(outcomes)
        n_deleted += deleted
        print(f"   {title}: {len(outcomes)} flagged -- {fixed} corrected, {deleted} deleted, "
              f"{len(outcomes) - fixed - deleted} kept")
        log.append(f"## {title}\n\n" + ("\n".join(f"- {o}" for o in outcomes) or "- Nothing flagged"))
        checked.append((title, new))
    whole = "\n\n".join(t for _, t in checked)
    print(f"   Independent audit of {AUDIT_SAMPLE} sentences against their full sources...")
    results = audit(whole, by_idx, api_key, model)
    v = make_verdict(results, n_reviewed, n_deleted, not_checked)
    log.append("## Independent audit\n\n" + "\n".join(
        f"- {verdict_}: {sentence}\n  → {reason}" for sentence, verdict_, reason in results))
    return checked, log, v


def split_brief(brief):
    """A brief's (header, [(title, text)] sections, references) parts."""
    body, _, refs = brief.partition("\n## References")
    chunks = re.split(r"^## ", body, flags=re.M)
    header = chunks[0]
    sections = []
    for chunk in chunks[1:]:
        title, _, text = chunk.partition("\n")
        sections.append((title.strip(), text.strip().rstrip("-").strip()))
    return header, sections, ("## References" + refs) if refs else ""


def run_verify(args):
    """
    python agent.py verify [brief.md] -- run the fact-check gate on a brief
    (by default the newest brief_*.md), e.g. after a teammate edited it.
    Writes <brief>_verified.md (corrected) and <brief>_verify_report.md; the
    original is not changed.
    """
    import glob
    import json
    import os
    import sys
    from synthesize import _load_all_evidence, _load_corpus

    path = args.question or (sorted(glob.glob("brief_*.md")) or [None])[-1]
    path = path if path and not path.endswith(("_verified.md", "_verify_report.md")) else None
    if not path or not os.path.exists(path):
        sys.exit("No brief found -- run synthesize first, or pass a file: python agent.py verify brief.md")
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("No API key found -- run install.py to add one (or pass --key KEY)")

    print("=" * 60 + f"\nVERIFY: {path}\n" + "=" * 60)
    by_idx = {int(d["idx"]): d for d in _load_corpus(args.corpus)}
    header, sections, refs = split_brief(open(path, encoding="utf-8").read())
    checked, log, v = gate(sections, by_idx, _load_all_evidence(args.corpus), api_key)

    stem = path[:-3]
    with open(stem + "_verified.md", "w", encoding="utf-8") as f:
        f.write(header.rstrip() + f"\n\n**Fact-check gate:** {verdict_line(v)}\n\n---\n\n"
                + "\n\n---\n\n".join(f"## {t}\n\n{c}" for t, c in checked) + "\n\n" + refs)
    with open(stem + "_verify_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Fact-check gate: {verdict_line(v)}\n\n" + "\n\n".join(log))
    with open(os.path.join(args.corpus, "brief_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2)
    print(f"\n{'=' * 60}\n{verdict_line(v)}\n   {stem}_verified.md\n   {stem}_verify_report.md\n{'=' * 60}")
