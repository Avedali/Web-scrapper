"""verify.py: checks that keep the brief tied to its sources.
Run: python agent.py verify [brief.md]

How it works:
  Quote check: quote_in_source() finds a quote in its source, ignoring formatting
  Local judge: judge_support() tests by meaning (NLI, 100+ languages) whether a passage supports a claim
  Gate: rules, the local judge, an AI review and an independent audit decide PASS or NEEDS REVIEW"""

import re
import unicodedata
from difflib import SequenceMatcher


# ════════════════════════════════════════════════════════════════════════
# A. IS THE QUOTE IN THE SOURCE?
# ════════════════════════════════════════════════════════════════════════

QUOTE_MIN_CHARS = 15     # Shortest quote that counts
QUOTE_MATCH = 0.9        # Share of the quote that must match, in order


def normalize(text):
    """Return lowercase letters and numbers only, without quotes, dashes, soft hyphens or line-break splits."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[^\w%$€£¥.,]+", " ", text.lower())
    text = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", text)  # Keep . and , only inside numbers
    return re.sub(r"\s+", " ", text).strip()


def quote_in_source(quote, source):
    """True if `quote` is in `source` after normalizing, or QUOTE_MATCH of it matches in order."""
    q, s = normalize(quote), normalize(source)
    if len(q) < QUOTE_MIN_CHARS:
        return False
    if q in s:
        return True
    # Fuzzy match only around places where 20-character pieces of the quote occur
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

NLI_MODEL = "joeddav/xlm-roberta-large-xnli"   # Same model as analyze
SUPPORTED = 0.5        # Entailment probability for "supported"
CONTRADICTED = 0.5     # Contradiction probability for "contradicted"

_nli = {}   # Loaded model cache


def judge_available():
    """True if PyTorch and transformers are installed for the local judge."""
    if "available" not in _nli:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _nli["available"] = True
        except Exception:
            _nli["available"] = False
            print("   (local judge unavailable -- PyTorch/transformers not installed; "
                  "the other checks still run)")
    return _nli["available"]


def _load_nli():
    """Load the NLI model once, on the Apple GPU (mps) when available."""
    if "model" not in _nli:
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
    """Return {"entailment": p, "neutral": p, "contradiction": p} for each (premise, hypothesis) pair."""
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
            # GPU failed: fall back to CPU for the rest of the run
            nli["model"].to("cpu")
            nli["device"] = "cpu"
            with torch.no_grad():
                logits = nli["model"](**enc).logits
        for row in logits.softmax(-1).cpu().tolist():
            results.append({nli["labels"][j]: p for j, p in enumerate(row)})
    return results


def verdict(probs):
    """Return 'supported', 'contradicted' or 'not shown' for one probability dict."""
    if probs["entailment"] >= SUPPORTED and probs["entailment"] > probs["contradiction"]:
        return "supported"
    if probs["contradiction"] >= CONTRADICTED:
        return "contradicted"
    return "not shown"


def judge_support(premise, hypothesis):
    """Return the verdict for one (premise, hypothesis) pair."""
    return verdict(judge_support_many([(premise, hypothesis)])[0])


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYERS 1-2: RULES + LOCAL JUDGE
# ════════════════════════════════════════════════════════════════════════

MAX_CITES = 3            # Max sources per claim
PASSAGES = 5             # Source sentences shown to the judge per claim
MATCH_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Passage matching


def sentences(text):
    """Split text into sentences; each line is split separately."""
    out = []
    for line in text.split("\n"):
        out += [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", line) if x.strip()]
    return out


def citations(sentence):
    """Return the [n] source numbers a sentence cites."""
    return [int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence)
            for n in re.split(r"\s*,\s*", group)]


def _readings(raw):
    """Return the possible readings of a written number: "1.500" gives {"1.5", "1500"}."""
    n = raw.replace("\u00a0", " ")
    out = set()
    if re.fullmatch(r"\d{1,3}(?:[, .]\d{3})+", n) and not n.startswith("0"):
        out.add(re.sub(r"[, .]", "", n))
    if re.fullmatch(r"\d+[.,]\d+", n) and not (out and "," in n):
        out.add(n.replace(",", "."))
    if not out:
        out.add(re.sub(r"[ ,]", "", n) if re.fullmatch(r"[\d ,]+", n) else n.replace(",", "."))
    return {x.rstrip("0").rstrip(".") if "." in x else x for x in out}


def number_readings(text):
    """Return {raw number: readings} for a text, skipping single digits and [n] citations."""
    text = unicodedata.normalize("NFKC", re.sub(r"\[[\d,\s]+\]", " ", text or ""))
    found = {}
    for raw in re.findall(r"\d+(?:[.,\u00a0 ]\d+)*", text):
        readings = _readings(raw)
        if max(len(x.replace(".", "")) for x in readings) >= 2:
            found[raw.strip()] = readings
    return found


def numbers(text):
    """Return every reading of every number in a text as one set."""
    return set().union(*number_readings(text).values()) if text else set()


_matcher = {}


def _match_model():
    """Return the sentence-matching model, or None if sentence-transformers is missing."""
    if "model" not in _matcher:
        try:
            from sentence_transformers import SentenceTransformer
            _matcher["model"] = SentenceTransformer(MATCH_MODEL)
        except Exception:
            _matcher["model"] = None
    return _matcher["model"]


def best_passages(claim, source_texts, k=PASSAGES):
    """Return the k source sentences closest to `claim`, by meaning or else by shared words and numbers."""
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
    return list(dict.fromkeys(candidates[i] for i in top))[:k]


def statements(text):
    """Return sentences that state something, skipping headings, tables and fragments."""
    return [s for s in sentences(text)
            if not s.startswith(("#", "**", "---", "|")) and len(re.findall(r"\w+", s)) >= 4]


def check_sentences(text, by_idx, findings="", use_judge=True):
    """Return {"sentence", "cites", "problems", "passages"} per statement: cite, number, repeat and judge checks."""
    use_judge = use_judge and judge_available()
    findings_numbers = numbers(findings)
    seen, reports, to_judge = set(), [], []
    for s in statements(text):
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
            # Judge each cited part against its own sources
            for part, part_cites in claim_parts(s):
                texts = [by_idx[n].get("content", "") for n in part_cites if n in by_idx]
                passages = best_passages(part, texts)
                if passages:
                    report["passages"] += passages
                    to_judge.append((report, part, passages))

    # Batched judge run; a sentence gets its weakest part's verdict
    if to_judge:
        verdicts = judge_with_passages([(part, passages) for _, part, passages in to_judge])
        order = {"supported": 0, "not shown": 1, "contradicted": 2}
        for (report, _, _), v in zip(to_judge, verdicts):
            if order[v] >= order[report.get("judge", "supported")]:
                report["judge"] = v
        for report in reports:
            if report.get("judge", "supported") != "supported":
                report["problems"].append(f"local judge: {report['judge']} by the cited passages")
    return reports


def judge_with_passages(items):
    """Return a verdict per (claim, passages) item; contradiction counts only from the best passage."""
    pairs, spans = [], []
    for claim, passages in items:
        options = list(dict.fromkeys(passages))
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
        elif mine[0]["contradiction"] >= CONTRADICTED:
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
    r"(?:an?\s+)?(?:[\w&.'’\-]+\s+){0,4}?(?:reports?|reported|says|said|found|finds|"
    r"estimates|claims|notes|states|indicated|indicates|showed|shows)\s+(?:that\s+)?",
]


def core_claim(claim):
    """Return the claim without lead-ins like "another study indicated that" (they confuse the judge)."""
    text = claim.split("; ")[-1].strip()
    for _ in range(3):
        before = text
        for pattern in _LEAD_INS:
            text = re.sub(r"^" + pattern, "", text, flags=re.IGNORECASE).strip()
        if text == before:
            break
    return text[:1].upper() + text[1:] if len(re.findall(r"\w+", text)) >= 3 else claim


def claim_parts(sentence):
    """Split a sentence at its citations into (text, [cited sources]) parts."""
    parts, start = [], 0
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence):
        text = sentence[start:m.start()].strip(" ,;:")
        cites = [int(n) for n in re.split(r"\s*,\s*", m.group(1))]
        if len(re.findall(r"\w+", text)) >= 3:
            parts.append((text, cites))
        elif parts:
            parts[-1] = (parts[-1][0], parts[-1][1] + cites)
        start = m.end()
    return parts


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYER 3: AI REVIEW AND REPAIR
# ════════════════════════════════════════════════════════════════════════

CHECK_MODEL = "google/gemini-2.5-flash"   # Reviewer model, stronger than the writer

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
    """Return the AI review prompt for one section's sentences."""
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
        "the cited facts, or a view or recommendation that starts 'Our view:' or 'At stake:' (these "
        "carry the argument -- keep them, unless they state a new fact); or this study's own results "
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
    """Gate one section; return (new text, outcome lines), or (text, None) if the AI call failed."""
    from core import call_ai

    reports = check_sentences(text, by_idx, findings, use_judge=use_judge)
    # Repeats are deleted without an AI call
    outcomes = []
    for r in reports:
        if "repeats an earlier sentence" in r["problems"]:
            at = text.rfind(r["sentence"])
            if at != -1:
                text = text[:at] + text[at + len(r["sentence"]):]
            outcomes.append(f"REPEAT: {r['sentence']}\n  → DELETED")
    items = [r for r in reports if "repeats an earlier sentence" not in r["problems"]
             and (r["problems"] or not r["cites"])]
    if not items:
        return text, outcomes

    def ask(batch):
        """Return {1-based position: (action, fix)} from one review call, or None if it failed."""
        reply = call_ai(_review_prompt(batch, findings), api_key, model, temp=0, max_tokens=12000)
        if not reply:
            return None
        got = {}
        for m in re.finditer(r"^\s*\**S(\d+)\**\s*:\s*(KEEP|DELETE|FILLER|FIX\s*:\s*(.+))\s*$", reply, re.M):
            got[int(m.group(1))] = (m.group(2).split(":")[0].strip().upper(), (m.group(3) or "").strip())
        return got

    answers = ask(items)
    if answers is None:
        return text, None
    # Retry skipped sentences once
    unanswered = [i for i in range(1, len(items) + 1) if i not in answers]
    if unanswered:
        again = ask([items[i - 1] for i in unanswered]) or {}
        for pos, i in enumerate(unanswered, 1):
            if pos in again:
                answers[i] = again[pos]

    for i, it in enumerate(items, 1):
        action, fix = answers.get(i, ("NO ANSWER", ""))
        hard = [p for p in it["problems"] if not p.startswith("local judge")]
        label = "; ".join(it["problems"]) or "uncited"
        if action == "KEEP" and not hard:
            if it["problems"]:
                outcomes.append(f"KEPT ({label}): {it['sentence']}\n  → the AI judged it supported")
            continue
        if action == "FIX" and fix:
            # A fix must pass the rules and not be contradicted by the judge
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
            continue
        else:
            outcomes.append(f"{action} ({label}): {it['sentence']}\n  → DELETED")
        text = text.replace(it["sentence"], "", 1)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), outcomes


# ════════════════════════════════════════════════════════════════════════
# THE GATE, LAYER 4: AUDIT AND VERDICT
# ════════════════════════════════════════════════════════════════════════

AUDIT_SAMPLE = 20        # Sentences re-checked against full sources
PASS_UNSUPPORTED = 0.10  # Max unsupported share of the audit sample
PASS_DELETED = 0.20      # Max deleted share of the draft's statements


def audit(text, by_idx, api_key, model=CHECK_MODEL, n=AUDIT_SAMPLE, seed=0):
    """Return (sentence, SUPPORTED/PARTLY/UNSUPPORTED/ERROR, reason) for a random sample checked against full sources."""
    import random
    from concurrent.futures import ThreadPoolExecutor
    from core import call_ai

    cited = [s for s in statements(text) if citations(s)]
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


def make_verdict(audit_results, n_reviewed, n_deleted, not_checked, n_statements=0):
    """Return the PASS / NEEDS REVIEW verdict dict with its reasons and counts."""
    graded = [v for _, v, _ in audit_results if v != "ERROR"]
    counts = {k: graded.count(k) for k in ("SUPPORTED", "PARTLY", "UNSUPPORTED")}
    reasons = []
    if not_checked:
        reasons.append(f"{len(not_checked)} section(s) not fact-checked: {', '.join(not_checked)}")
    if not graded:
        reasons.append("the independent audit could not run")
    elif counts["UNSUPPORTED"] / len(graded) > PASS_UNSUPPORTED:
        reasons.append(f"audit: {counts['UNSUPPORTED']} of {len(graded)} sampled sentences unsupported")
    total = n_statements or n_reviewed
    if total and n_deleted / total > PASS_DELETED:
        reasons.append(f"the gate deleted {n_deleted} of the draft's {total} statements")
    return {
        "status": "NEEDS REVIEW" if reasons else "PASS",
        "reasons": reasons,
        "audit": counts, "audit_sample": len(graded),
        "reviewed": n_reviewed, "deleted": n_deleted, "statements": total,
    }


def verdict_line(v):
    """Return the verdict as one line for the brief header and console."""
    a, n = v["audit"], v["audit_sample"]
    audit_text = (f"independent audit of {n} sentences: {a['SUPPORTED']} supported, "
                  f"{a['PARTLY']} partly, {a['UNSUPPORTED']} unsupported") if n else "no audit"
    of = f" of {v['statements']} statements" if v.get("statements") else ""
    return (f"{v['status']}. {audit_text[:1].upper() + audit_text[1:]}; the gate reviewed "
            f"{v['reviewed']} sentences and deleted {v['deleted']}{of}."
            + (f" Why: {'; '.join(v['reasons'])}." if v["reasons"] else ""))


# ════════════════════════════════════════════════════════════════════════
# THE WHOLE GATE AND "python agent.py verify"
# ════════════════════════════════════════════════════════════════════════

def gate(sections, by_idx, findings, api_key, model=CHECK_MODEL, run_audit=True):
    """Return (checked sections, log lines, verdict) for [(title, text)] sections; run_audit=False skips the audit."""
    checked, log, not_checked, n_reviewed, n_deleted = [], [], [], 0, 0
    n_statements = sum(len(statements(text)) for _, text in sections)
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
        print(f"   {title}: {len(outcomes)} flagged: {fixed} corrected, {deleted} deleted, "
              f"{len(outcomes) - fixed - deleted} kept")
        log.append(f"## {title}\n\n" + ("\n".join(f"- {o}" for o in outcomes) or "- Nothing flagged"))
        checked.append((title, new))
    results = []
    if run_audit:
        whole = "\n\n".join(t for _, t in checked)
        print(f"   Independent audit of {AUDIT_SAMPLE} sentences against their full sources...")
        results = audit(whole, by_idx, api_key, model)
        log.append("## Independent audit\n\n" + "\n".join(
            f"- {verdict_}: {sentence}\n  → {reason}" for sentence, verdict_, reason in results))
    v = make_verdict(results, n_reviewed, n_deleted, not_checked, n_statements)
    v["not_checked"] = not_checked
    return checked, log, v


def merge_verdicts(main, extra):
    """Combine the main verdict (with audit) and a later unaudited one into one verdict."""
    graded = main["audit"]
    fake_audit = [("", k, "") for k in ("SUPPORTED", "PARTLY", "UNSUPPORTED") for _ in range(graded[k])]
    v = make_verdict(fake_audit, main["reviewed"] + extra["reviewed"],
                     main["deleted"] + extra["deleted"],
                     main.get("not_checked", []) + extra.get("not_checked", []),
                     main.get("statements", 0) + extra.get("statements", 0))
    v["not_checked"] = main.get("not_checked", []) + extra.get("not_checked", [])
    return v


def split_brief(brief):
    """Return a brief's (header, [(title, text)] sections, references)."""
    body, _, refs = brief.partition("\n## References")
    chunks = re.split(r"^## ", body, flags=re.M)
    header = chunks[0]
    sections = []
    for chunk in chunks[1:]:
        title, _, text = chunk.partition("\n")
        sections.append((title.strip(), text.strip().rstrip("-").strip()))
    return header, sections, ("## References" + refs) if refs else ""


def run_verify(args):
    """Gate a brief (default: newest brief_*.md); writes <brief>_verified.md and <brief>_verify_report.md."""
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
