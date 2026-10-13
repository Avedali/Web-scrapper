"""
explore.py: Mode 1.a, exploratory research. Gathers sources; forms no thesis.
Run: python agent.py explore --key KEY (or launcher.py, option 1)
How it works:
  Define: quotes dictionary definitions of the topic's key terms
  Decompose: native search queries in 9 languages
  Search: web search per query, pages read in parallel
  Authoritative/Experiential: surveys, papers, forums and blogs
  Red team + bias check: AI critiques the corpus; its fix queries fill gaps
Saves scratchpad/corpus.json and sources_meta.json.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import re
import json
import sys
import os

from collections import defaultdict

from core import (
    search_many,
    region_for,
    fetch_pages,
    PAGE_META,
    CONTENT_CHARS,
    search_academic,
    search_hn,
    call_ai,
    pad,
    classify_source,
    detect_lang,
    extract_search_queries,
    QUERY_FORMAT,
    init_search,
    print_capabilities,
    LANGUAGES,
    SCRATCHPAD,
)

# ════════════════════════════════════════════════════════════════════════
# SETTINGS: search depth and default topic
# ════════════════════════════════════════════════════════════════════════

QUERIES_PER_LANGUAGE = 8   # Native search queries per language
RESULTS_PER_QUERY = 5      # Search results read per query
GAP_QUERIES = 12           # Fix-up searches suggested by red team + bias check


DEFAULT_TOPIC = (
    "How AI-powered coding assistants (GitHub Copilot, Cursor, ChatGPT, "
    "Claude Code, Windsurf) are transforming the software development "
    "industry: productivity, workforce, business models, and risks"
)


# ════════════════════════════════════════════════════════════════════════
# KEY TERMS: definitions quoted from reference works, not written by the AI
# ════════════════════════════════════════════════════════════════════════

REFERENCE_DOMAINS = [
    "merriam-webster.com", "dictionary.cambridge.org", "oxfordlearnersdictionaries.com",
    "oed.com", "collinsdictionary.com", "dictionary.com", "britannica.com",
    "wiktionary.org", "thefreedictionary.com",
    "wikipedia.org", "csrc.nist.gov", "iso.org", "gartner.com/en/information-technology/glossary",
    "techtarget.com", "investopedia.com", "oecd.org",
]
DEFINITIONS_PER_TERM = 4   # Quoted definitions kept per term (one per site)
MAX_TERMS = 12             # Terms looked up


# Sales lines at the end of a dictionary snippet
_SITE_TAIL = re.compile(r"(\s*(Learn more|See examples of .*? used in a sentence|How to use .*? in a "
                        r"sentence|See more|Read more)\.?(\.\.\.)?)+\s*$", re.I)


def _reference_site(url):
    """Return the reference domain a URL belongs to, or None."""
    u = url.lower().split("://")[-1]
    return next((d for d in REFERENCE_DOMAINS if u.startswith(d) or f".{d}" in u.split("/")[0]
                 or ("/" in d and d in u)), None)


def define_terms(topic, api_key, model):
    """Return Markdown of quoted reference definitions per key term, topic-sense first."""
    reply = call_ai(
        "List the key terms of this research topic: the words and phrases a reader must "
        "understand to research it. Write each as it would appear as a dictionary or "
        "encyclopedia entry (e.g. 'machine learning', not 'ML-driven'). For a phrase that is "
        "not a common entry, also list its parts that are. One term per line. Do not define "
        f"them.\n\nTOPIC: {topic}",
        api_key, model, temp=0)
    terms = []
    for line in (reply or "").splitlines():
        t = re.sub(r"^[\s\-*•\d.)]+", "", line).strip().strip('"*').strip()
        if 2 <= len(t) <= 60 and t.lower() not in {x.lower() for x in terms}:
            terms.append(t)
    terms = terms[:MAX_TERMS]

    # One result per reference site; terms with none are retried unquoted
    def lookup(results):
        kept, sites = [], set()
        for r in results or []:
            site = _reference_site(r.get("url", ""))
            text = re.sub(r"\s+", " ", r.get("snippet", "")).strip()
            text = _SITE_TAIL.sub("", text).strip()
            if site and site not in sites and len(text) > 20:
                sites.add(site)
                kept.append({"site": site, "text": text, "url": r["url"]})
        return kept[:DEFINITIONS_PER_TERM]

    found = search_many([(f'"{t}" definition', 10, "us-en") for t in terms])
    senses = {t: lookup(r) for t, r in zip(terms, found)}
    missing = [t for t in terms if not senses[t]]
    for t, r in zip(missing, search_many([(f"{t} definition", 10, "us-en") for t in missing])):
        senses[t] = lookup(r)

    # AI picks, by number, the definitions matching the topic's sense
    numbered, ref = [], {}
    for term, kept in senses.items():
        for d in kept:
            ref[str(len(ref) + 1)] = (term, d)
            numbered.append(f"D{len(ref)} [{term}] {d['text']}")
    match = set()
    if numbered:
        pick = call_ai(
            f"TOPIC: {topic}\n\nBelow are dictionary definitions of the topic's key terms. "
            "List the numbers of the definitions that match how the term is used in the topic, "
            "e.g. 'D1, D4, D5'. Nothing else.\n\n" + "\n".join(numbered),
            api_key, model, temp=0) or ""
        match = set(re.findall(r"D\s*(\d+)", pick))
        if not match:                    # Failed call: keep every definition
            match = set(ref)

    out = ["Each definition is quoted from the reference work named, as its search "
           "result shows it. None is written by the AI.\n"]
    for term, kept in senses.items():
        out.append(f"## {term}\n")
        if not kept:
            out.append("No dictionary or reference definition found.\n")
            continue
        nums = [n for n, (t, _) in ref.items() if t == term]
        fit = [ref[n][1] for n in nums if n in match]
        other = [ref[n][1] for n in nums if n not in match]
        for d in fit:
            out.append(f"- \"{d['text']}\" ({d['site']}, {d['url']})")
        if other:
            out.append("\nOther senses:")
            out += [f"- \"{d['text']}\" ({d['site']}, {d['url']})" for d in other]
        out.append("")
    found_n = sum(1 for k in senses.values() if k)
    print(f"   {found_n} of {len(terms)} terms found in dictionaries or reference works")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════
# SEARCH QUERY PROMPTS: the same neutral request in each language
# ════════════════════════════════════════════════════════════════════════

QUERY_PROMPTS = {
    "en": ("Write {n} web search queries to research this topic. Keep them neutral: each should "
           "look for evidence (for example \"effect of X on Y\"), not assume a conclusion (for "
           "example \"how X improves Y\"). Cover different aspects of the topic. Write them the "
           "way people type them into a search engine.\nOutput exactly {n} lines, one query per "
           "line, with no numbering and no commentary.\n\nTOPIC: {topic}"),
    "fr": ("Rédigez {n} requêtes de recherche web pour étudier ce sujet. Restez neutre : chaque "
           "requête doit chercher des données (par exemple « effet de X sur Y »), sans présupposer "
           "de conclusion (par exemple « comment X améliore Y »). Couvrez différents aspects du "
           "sujet. Écrivez-les comme on les tape dans un moteur de recherche.\nRépondez par "
           "exactement {n} lignes, une requête par ligne, sans numérotation ni commentaire."
           "\n\nSUJET : {topic}"),
    "de": ("Schreiben Sie {n} Suchanfragen für eine Websuche, um dieses Thema zu recherchieren. "
           "Bleiben Sie neutral: Jede Anfrage soll nach Belegen suchen (zum Beispiel „Auswirkung "
           "von X auf Y“), ohne ein Ergebnis vorwegzunehmen (zum Beispiel „wie X Y verbessert“). "
           "Decken Sie verschiedene Aspekte des Themas ab. Formulieren Sie die Anfragen so, wie man "
           "sie in eine Suchmaschine eingibt.\nAntworten Sie mit genau {n} Zeilen, eine "
           "Suchanfrage pro Zeile, ohne Nummerierung und ohne Kommentar.\n\nTHEMA: {topic}"),
    "ja": ("このテーマを調べるためのウェブ検索クエリを{n}個作成してください。中立を保ってください。"
           "各クエリは根拠となるデータを探すものとし（例：「XがYに与える影響」）、結論を前提にしないで"
           "ください（例：「XはどのようにYを改善するか」）。テーマのさまざまな側面を取り上げてください。"
           "検索エンジンに実際に入力する形で書いてください。\nちょうど{n}行で答えてください。"
           "1行に1つのクエリとし、番号や説明は付けないでください。\n\nテーマ：{topic}"),
    "ko": ("이 주제를 조사하기 위한 웹 검색어를 {n}개 작성하세요. 중립을 지키세요. 각 검색어는 근거 "
           "자료를 찾는 것이어야 하며(예: 'X가 Y에 미치는 영향'), 결론을 전제해서는 안 됩니다(예: "
           "'X는 어떻게 Y를 개선하는가'). 주제의 여러 측면을 다루세요. 검색 엔진에 실제로 입력하는 "
           "형태로 쓰세요.\n정확히 {n}줄로 답하세요. 한 줄에 검색어 하나씩, 번호나 설명 없이 "
           "쓰세요.\n\n주제: {topic}"),
    "zh": ("请为研究这个主题写出{n}个网络搜索查询。保持中立：每个查询都应寻找证据（例如“X对Y的影响”），"
           "而不是预设结论（例如“X如何改善Y”）。涵盖该主题的不同方面。按照人们在搜索引擎中实际输入的"
           "方式来写。\n请恰好回答{n}行，每行一个查询，不要编号，也不要任何说明。\n\n主题：{topic}"),
    "pt": ("Escreva {n} consultas de pesquisa na web para investigar este tema. Mantenha a "
           "neutralidade: cada consulta deve buscar evidências (por exemplo, \"efeito de X sobre "
           "Y\"), sem pressupor uma conclusão (por exemplo, \"como X melhora Y\"). Cubra "
           "diferentes aspectos do tema. Escreva as consultas do jeito que as pessoas digitam num "
           "buscador.\nResponda com exatamente {n} linhas, uma consulta por linha, sem numeração "
           "e sem comentários.\n\nTEMA: {topic}"),
    "es": ("Escribe {n} consultas de búsqueda web para investigar este tema. Mantén la "
           "neutralidad: cada consulta debe buscar datos (por ejemplo, «efecto de X en Y») sin dar "
           "por hecha una conclusión (por ejemplo, «cómo X mejora Y»). Cubre distintos aspectos "
           "del tema. Escríbelas como se escriben en un buscador.\nResponde con exactamente {n} "
           "líneas, una consulta por línea, sin numeración ni comentarios.\n\nTEMA: {topic}"),
    "ru": ("Составьте {n} поисковых запросов, чтобы исследовать эту тему. Сохраняйте "
           "нейтральность: каждый запрос должен искать данные (например, «влияние X на Y»), а не "
           "предполагать вывод (например, «как X улучшает Y»). Охватите разные аспекты темы. "
           "Пишите запросы так, как их вводят в поисковую систему.\nОтветьте ровно {n} строками, "
           "по одному запросу в строке, без нумерации и комментариев.\n\nТЕМА: {topic}"),
}


# ════════════════════════════════════════════════════════════════════════
# QUERY PARSING: AI reply to clean search queries
# ════════════════════════════════════════════════════════════════════════

# Script a real query must contain (ja, ko, zh, ru)
NATIVE_SCRIPT = {
    "ja": r"[\u3040-\u30ff\u4e00-\u9fff]",
    "ko": r"[\uac00-\ud7a3]",
    "zh": r"[\u4e00-\u9fff]",
    "ru": r"[\u0400-\u04ff]",
}

# A query marked up as **bold**, "quoted", “curly” or 「Japanese」
_MARKED = re.compile(r'\*\*(.+?)\*\*|"([^"]{4,})"|\u201c([^\u201d]{4,})\u201d|\u300c([^\u300d]{4,})\u300d')


def _parse_queries(resp, n, lang=None):
    """Return at most n queries from an AI reply, dropping preambles, translations and commentary."""
    lines = []
    for raw_line in (resp or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-•*])\s+", "", raw_line).strip()
        m = _MARKED.match(line)
        lines.append((line, next(g for g in m.groups() if g) if m else None))

    # If any line is marked up, only marked text counts
    if any(marked for _, marked in lines):
        lines = [(marked, marked) for _, marked in lines if marked]

    queries = []
    for line, _ in lines:
        q = line.strip(' *"\u201c\u201d\u300c\u300d')
        if len(q) <= (3 if lang in ("ja", "ko", "zh") else 8):
            continue
        if re.fullmatch(r"[(\uff08].*[)\uff09]", q):
            continue
        if q.endswith((":", "\uff1a")):
            continue
        if q.lower().startswith(("translation", "here are", "these ", "this query", "note")):
            continue
        if lang in NATIVE_SCRIPT and not re.search(NATIVE_SCRIPT[lang], q):
            continue
        queries.append(q)

    return queries[:n]


# ════════════════════════════════════════════════════════════════════════
# GATHERING: search, then read new pages in parallel
# ════════════════════════════════════════════════════════════════════════

def _gather(jobs, sources, quality=None):
    """Search (query, n, region, search_lang) jobs, add new readable pages to sources, return count."""
    results = search_many([(q, n, region) for q, n, region, _ in jobs])

    todo = {}
    for job, found in zip(jobs, results, strict=True):
        for r in found:
            if r["url"] not in sources and r["url"] not in todo:
                todo[r["url"]] = (job, r)

    pages = fetch_pages(list(todo))

    added = 0
    for url, ((q, _, _, search_lang), r) in todo.items():
        content = pages.get(url)
        if not content:
            continue
        sources[url] = {
            "title": r["title"],
            "domain": r["domain"],
            "snippet": r.get("snippet", ""),
            "content": content,
            "queries": [q],
            "quality": quality or classify_source(r["domain"]),
            "source_lang": detect_lang(content),
            "search_lang": search_lang,
            **PAGE_META.get(url, {}),
        }
        added += 1
    return added


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def run_explore(args):
    """Run all explore phases and save the corpus to the scratchpad."""

    # ── Setup ───────────────────────────────────────────────────────
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("No API key found -- run install.py to add one (or pass --key KEY)")

    model = args.model
    topic = args.topic or DEFAULT_TOPIC
    init_search(args.serper, args.brave)

    print("=" * 60)
    print("MODE 1.a: EXPLORE (no opinion, just gather)")
    print_capabilities()
    print(f"   Topic: {topic[:60]}...")
    print("=" * 60)

    sources = {}

    # ── Phase 1: Define key terms ───────────────────────────────────
    print(f"\n{'─'*50}\nDEFINE\n{'─'*50}")

    defs = define_terms(topic, api_key, model)

    pad("01_definitions", f"# Definitions\n\n{defs}")

    # ── Phase 2: Native decomposition (9 languages) ─────────────────
    print(f"\n{'─'*50}\nDECOMPOSE ({len(LANGUAGES)} languages)\n{'─'*50}")

    all_queries = {}

    # Topic translated per language, kept in the file for checking
    topics = {"en": topic}
    for lc, info in LANGUAGES.items():
        if lc != "en":
            t = call_ai(f"Translate this research topic into {info['name']}. Keep product and "
                        f"company names as they are. Output only the translation.\n\n{topic}",
                        api_key, model, temp=0).strip()
            topics[lc] = t if t and (lc not in NATIVE_SCRIPT or re.search(NATIVE_SCRIPT[lc], t)) else topic

    # Prompt written in each language; one retry if nothing parses
    for lc, info in LANGUAGES.items():
        n_queries = QUERIES_PER_LANGUAGE

        prompt = QUERY_PROMPTS[lc].format(n=n_queries, topic=topics[lc])

        resp = call_ai(prompt, api_key, model, temp=0.5)
        qs = _parse_queries(resp, n_queries, lc)

        if not qs:
            resp = call_ai(prompt, api_key, model, temp=0.7)
            qs = _parse_queries(resp, n_queries, lc)

        all_queries[lc] = qs
        print(f"   {info['name']}: {len(qs)} queries")

    doc = "# Native Decomposition\n\n"
    for lc, qs in all_queries.items():
        doc += f"Topic as given to the {LANGUAGES[lc]['name']} prompt: {topics[lc]}\n\n" if lc != "en" else ""
        doc += f"## {LANGUAGES[lc]['name']}\n" + "\n".join(f"- {q}" for q in qs) + "\n\n"
    pad("02_decomposition", doc)

    # ── Phase 3: Web search sweep ───────────────────────────────────
    print(f"\n{'─'*50}\nSEARCH\n{'─'*50}")
    print(f"   {sum(len(q) for q in all_queries.values())} queries x {RESULTS_PER_QUERY} results, "
          f"reading pages in parallel...")

    _gather([(q, RESULTS_PER_QUERY, LANGUAGES[lc]["region"], lc)
             for lc, queries in all_queries.items() for q in queries], sources)

    for lc in all_queries:
        n = sum(1 for s in sources.values() if s["search_lang"] == lc)
        print(f"   {LANGUAGES[lc]['name']:16s} {n} sources")

    # ── Phase 4: Authoritative and academic sources ─────────────────
    print(f"\n{'─'*50}\n AUTHORITATIVE\n{'─'*50}")

    added = _gather([(q, 3, region, "mixed") for q, region in [
        ("Stack Overflow developer survey AI tools 2025 2026", "us-en"),
        ("JetBrains developer ecosystem survey AI", "us-en"),
        ("OECD AI software workers report", "us-en"),
        ("EU AI Act software development", "us-en"),
        ("GitHub Copilot productivity Microsoft study", "us-en"),
        ("AIコーディングアシスタント 調査 日本", "jp-jp"),
        ("中国AI编程工具 开发者 报告", "cn-zh"),
    ]], sources)
    print(f"   {added} authoritative sources")

    # Semantic Scholar + arXiv; the abstract is the content
    for q in ["AI code generation productivity", "GitHub Copilot study", "LLM code quality"]:
        for p in search_academic(q, n=3):
            if p["url"] not in sources:
                sources[p["url"]] = {
                    "title": p["title"],
                    "domain": p["domain"],
                    "snippet": p["snippet"],
                    "content": p["snippet"],
                    "queries": [q],
                    "quality": "academic",
                    "year": p.get("year"),
                    "authors": p.get("authors") or [],
                    "source_lang": "en",
                    "search_lang": "en",
                }

    # ── Phase 5: Experiential sources (developer voices) ────────────
    print(f"\n{'─'*50}\n EXPERIENTIAL\n{'─'*50}")

    for q in ["Copilot experience", "Cursor AI review", "coding assistant"]:
        for r in search_hn(q, n=5):
            if r["url"] not in sources:
                sources[r["url"]] = {
                    "title": r["title"],
                    "domain": r["domain"],
                    "snippet": r.get("snippet", ""),
                    "content": r.get("snippet", ""),
                    "queries": [q],
                    "quality": "experiential/forum",
                    "source_lang": "en",
                    "search_lang": "en",
                }

    added = _gather([(q, 5, region, "mixed") for q, region in [
        ("site:dev.to Copilot review developer", "us-en"),
        ('"mon expérience avec Copilot" développeur', "fr-fr"),
        ("Copilot 使ってみた エンジニア", "jp-jp"),
    ]], sources, quality="experiential/blog")
    print(f"   {added} developer blog posts")

    # ── Phase 6: Red team ───────────────────────────────────────────
    print(f"\n{'─'*50}\nRED TEAM\n{'─'*50}")

    lc = defaultdict(int)
    qc = defaultdict(int)

    for s in sources.values():
        lc[s.get("source_lang", "?")] += 1
        qc[s["quality"]] += 1

    critique = call_ai(
        f"Red-team this research. Look for: hallucination risks, SEO sources, "
        f"contradictions, gaps, single-source claims, stale data, vendor bias.\n\n"
        f"SOURCES: {len(sources)} | LANGS: {dict(lc)} | QUALITY: {dict(qc)}\n\n"
        + "\n".join(
            f"[{s['quality']}][{s.get('source_lang', '?')}] {s['title']}"
            for s in list(sources.values())[:25]
        )
        + "\n\nFor each gap, suggest a specific fix query. " + QUERY_FORMAT,
        api_key, model, temp=0.4,
        max_tokens=3000,   # Caps a model stuck repeating itself
    )

    pad("06_red_team", f"# Red Team\n\n{critique}")

    fixes = extract_search_queries(critique)

    # ── Phase 6.5: Bias check (geographic, optimism, demographic skew) ─
    print(f"\n{'─'*50}\n BIAS CHECK\n{'─'*50}")

    bias = call_ai(
        f"Bias audit. Check: geographic, optimism/pessimism, corporate, "
        f"demographic, temporal, linguistic.\nLANGS: {dict(lc)}\n"
        f"Suggest up to 12 fix queries (include non-English native script). " + QUERY_FORMAT,
        api_key, model, temp=0.4,
        max_tokens=3000,
    )

    pad("065_bias", f"# Bias Audit\n\n{bias}")

    fixes += [q for q in extract_search_queries(bias) if q not in fixes]

    # ── Phase 7: Gap fill (region matches each query's script) ──────
    print(f"\n{'─'*50}\nGAP FILL ({len(fixes[:GAP_QUERIES])} queries)\n{'─'*50}")

    added = _gather([(q, 3, region_for(q), "mixed") for q in fixes[:GAP_QUERIES]], sources)
    print(f"   {added} sources added")

    # ── Save: corpus.json and sources_meta.json, old ones backed up ─
    # Empty run: stop so a good corpus is not replaced
    if not sources:
        sys.exit("No sources found -- check the internet connection and try again. "
                 "Any previous corpus was kept.")

    import shutil
    from datetime import datetime

    corpus_path = os.path.join(SCRATCHPAD, "corpus.json")
    meta_path = os.path.join(SCRATCHPAD, "sources_meta.json")

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    for existing_file in [corpus_path, meta_path]:
        if os.path.exists(existing_file):
            name, ext = os.path.splitext(existing_file)
            backup = f"{name}-{timestamp}{ext}"
            shutil.copy2(existing_file, backup)
            print(f"   Backed up: {os.path.basename(backup)}")

    corpus = [
        {
            "idx": i + 1,
            "title": s["title"],
            "domain": s["domain"],
            "quality": s["quality"],
            "lang": s.get("source_lang", "?"),
            "url": url,
            "year": s.get("year"),
            "date": s.get("date", ""),
            "authors": s.get("authors") or [],
            "site": s.get("site", ""),
            "content": s["content"][:CONTENT_CHARS],
        }
        for i, (url, s) in enumerate(sources.items())
    ]

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    meta = {
        url: {k: v for k, v in s.items() if k != "content"}
        for url, s in sources.items()
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── Done ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXPLORATION COMPLETE -- {len(sources)} sources gathered")
    print("   No thesis yet. Run 'analyze' and 'stats' next,")
    print("   then 'synthesize' to let the data speak.")
    print(f"{'='*60}")
