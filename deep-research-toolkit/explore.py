"""
explore.py: Mode 1.a, exploratory research.
=====================================================
Run: python agent.py explore --key KEY
Or:  launcher.py → option 1

This is the first mode you run. It gathers raw material about the topic:
every relevant source it can find. It forms no opinion or thesis and
writes no sections.

It works in 7 phases:
  P1: Look up key terms in dictionaries (so we know what we're talking about)
  P2: Decompose into search queries (in 9 languages, phrased as native speakers would)
  P3: Search the web for each query and read the pages
  P4: Hunt for authoritative sources (surveys, academic papers)
  P5: Hunt for experiential sources (developer forums, blogs)
  P6: Red-team the corpus (find weaknesses, gaps, bad sources)
  P6.5: Bias check (whether the corpus is skewed geographically or ideologically)
  P7: Fill gaps identified by the red team and bias check

Everything is saved to the scratchpad/ folder for downstream modes.
The opinion comes later, from what the data shows (in synthesize mode).
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import re    # Pattern matching in text (used to clean up AI responses)
import json  # Read/write structured data files
import sys   # Stop with an error code when a step can't run
import os    # File/folder operations and environment variables

from collections import defaultdict # Auto-initializing dictionary (see core.py)

# Import our shared tools from core.py.
# "from core import X, Y, Z" means: bring these specific functions
# from core.py into this file so we can use them directly.
from core import (
    search_many,         # Run several web searches at once
    region_for,          # Search region matching a query's script (ja, ko, ...)
    fetch_pages,         # Read many web pages at once
    PAGE_META,           # Author, site and date of each page read
    CONTENT_CHARS,       # How much text to keep per source
    search_academic,     # Search Semantic Scholar + arXiv
    search_hn,           # Search Hacker News
    call_ai,             # Send a prompt to an AI model
    pad,                 # Save work to the scratchpad
    classify_source,     # Label a source's quality tier
    detect_lang,         # Detect what language a text is in
    extract_search_queries,  # Pull "QUERY: ..." lines out of AI text
    QUERY_FORMAT,            # How we ask the AI to mark its search queries
    init_search,         # Set up search engine(s)
    print_capabilities,  # Show which libraries are installed
    LANGUAGES,           # The 9 languages we search in
    SCRATCHPAD,          # Path to the scratchpad folder
)

# ════════════════════════════════════════════════════════════════════════
# DEFAULT TOPIC: used if the user doesn't specify one
# ════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════
# HOW DEEP TO SEARCH: raise for a bigger corpus (explore takes longer)
# ════════════════════════════════════════════════════════════════════════
#
# Was 5 queries x 3 results per language: at most 15 pages per language,
# ~10 after pages that can't be read. Now up to 40 per language. Searches
# and page downloads run in parallel, so a full explore takes about as long
# as the old, smaller one did.

QUERIES_PER_LANGUAGE = 8   # Native search queries per language
RESULTS_PER_QUERY = 5      # Search results read per query
GAP_QUERIES = 12           # Fix-up searches suggested by red team + bias check


DEFAULT_TOPIC = (
    "How AI-powered coding assistants (GitHub Copilot, Cursor, ChatGPT, "
    "Claude Code, Windsurf) are transforming the software development "
    "industry: productivity, workforce, business models, and risks"
)


# ════════════════════════════════════════════════════════════════════════
# KEY TERMS: definitions quoted from dictionaries, never written by the AI
# ════════════════════════════════════════════════════════════════════════
#
# A definition the AI writes from memory reflects its training data, and the
# framing of the brief then follows it. So the AI only lists the terms. Each
# term is searched with "definition", and only results from dictionaries and
# reference works are kept, quoted as the search engine shows them (for
# these sites that is the page's own definition), with the URL.

REFERENCE_DOMAINS = [
    # General dictionaries
    "merriam-webster.com", "dictionary.cambridge.org", "oxfordlearnersdictionaries.com",
    "oed.com", "collinsdictionary.com", "dictionary.com", "britannica.com",
    "wiktionary.org", "thefreedictionary.com",
    # Encyclopedias and technical glossaries
    "wikipedia.org", "csrc.nist.gov", "iso.org", "gartner.com/en/information-technology/glossary",
    "techtarget.com", "investopedia.com", "oecd.org",
]
DEFINITIONS_PER_TERM = 4   # Quoted definitions kept per term (one per site)
MAX_TERMS = 12             # Terms looked up


# The sites' own sales lines at the end of a search snippet
_SITE_TAIL = re.compile(r"(\s*(Learn more|See examples of .*? used in a sentence|How to use .*? in a "
                        r"sentence|See more|Read more)\.?(\.\.\.)?)+\s*$", re.I)


def _reference_site(url):
    """The reference work a URL belongs to ("merriam-webster.com"), or None."""
    u = url.lower().split("://")[-1]
    return next((d for d in REFERENCE_DOMAINS if u.startswith(d) or f".{d}" in u.split("/")[0]
                 or ("/" in d and d in u)), None)


def define_terms(topic, api_key, model):
    """
    Look up the topic's key terms in dictionaries and reference works.
    Returns the Markdown saved as scratchpad/01_definitions.md: for each term,
    the definitions that match the topic's sense, each quoted with its URL,
    then any other senses found. A term no reference work defines is listed
    as such, without a definition.
    """
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

    # Search each term; keep one result per reference site. A term with no
    # result is searched again without quotes (an exact phrase like "coding
    # assistant" can miss an entry titled differently).
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

    # The AI picks, by number, the quoted definitions that match the topic's
    # sense ("assistant" as software, not as a shop assistant). It chooses
    # among the quotes and writes nothing that goes in the file.
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
        if not match:                    # The call failed: keep every definition as matching
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
# SEARCH QUERY PROMPTS: the same request, written in each language
# ════════════════════════════════════════════════════════════════════════
# {n} is the number of queries and {topic} the topic in that language.
# Each asks for neutral queries ("effect of X on Y", not "how X improves Y")
# written the way people type them into a search engine.

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
# QUERY PARSING: turn the AI's answer into a clean list of search queries
# ════════════════════════════════════════════════════════════════════════
#
# Models rarely return bare queries. Real output looks like:
#   Here are 5 search queries in Spanish:            ← preamble
#   **"¿Revolución o reemplazo?"** (Translation: ...) ← query + translation
#   This query reflects concerns about jobs.          ← commentary
#   (The future of programming in Russia)             ← translation line
# Every one of those lines used to become a "query".

# Characters that must appear in a real query for non-Latin languages.
# A Korean "query" with no Hangul is English commentary, or comes from a
# model that dropped the native script (some providers do).
NATIVE_SCRIPT = {
    "ja": r"[\u3040-\u30ff\u4e00-\u9fff]",  # Hiragana, Katakana, Kanji
    "ko": r"[\uac00-\ud7a3]",                # Hangul
    "zh": r"[\u4e00-\u9fff]",                # Chinese characters
    "ru": r"[\u0400-\u04ff]",                # Cyrillic
}

# A query the model marked up: **bold**, "quoted", “curly” or 「Japanese」
_MARKED = re.compile(r'\*\*(.+?)\*\*|"([^"]{4,})"|\u201c([^\u201d]{4,})\u201d|\u300c([^\u300d]{4,})\u300d')


def _parse_queries(resp, n, lang=None):
    """Extract at most n search queries from an AI response."""
    lines = []
    for raw_line in (resp or "").splitlines():
        # Strip list markers: "1." "2)" "-" "•" "* ". A leading year and the
        # "**" of bold text stay (a marker needs a space after it).
        line = re.sub(r"^\s*(?:\d+[.)]|[-•*])\s+", "", raw_line).strip()
        # Only markup at the start of a line marks a query. A commentary
        # line that quotes a term ('such as "автокодирование"') and a
        # translation line ('(Translation: "...")') are skipped.
        m = _MARKED.match(line)
        lines.append((line, next(g for g in m.groups() if g) if m else None))

    # If the model marked its queries, only marked text counts. The
    # unmarked lines around them are explanations.
    if any(marked for _, marked in lines):
        lines = [(marked, marked) for _, marked in lines if marked]

    queries = []
    for line, _ in lines:
        q = line.strip(' *"\u201c\u201d\u300c\u300d')
        if len(q) <= (3 if lang in ("ja", "ko", "zh") else 8):   # CJK queries are short
            continue
        if re.fullmatch(r"[(\uff08].*[)\uff09]", q):        # "(translation)"
            continue
        if q.endswith((":", "\uff1a")):                     # "Here are 5 queries:"
            continue
        if q.lower().startswith(("translation", "here are", "these ", "this query", "note")):
            continue
        if lang in NATIVE_SCRIPT and not re.search(NATIVE_SCRIPT[lang], q):
            continue
        queries.append(q)

    return queries[:n]


# ════════════════════════════════════════════════════════════════════════
# GATHERING: search, then read the new pages, many at a time
# ════════════════════════════════════════════════════════════════════════

def _gather(jobs, sources, quality=None):
    """
    Run searches and add every readable page not already in `sources`.

    jobs: list of (query, n_results, region, search_lang). All searches run
    first (a few at a time), then all new pages are downloaded in parallel
    (see core.fetch_pages). quality overrides the domain-based label.
    Returns how many sources were added.
    """
    results = search_many([(q, n, region) for q, n, region, _ in jobs])

    # Collect the new URLs, remembering which query found each one
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
            continue  # Couldn't read the page
        sources[url] = {
            "title": r["title"],                     # Page title
            "domain": r["domain"],                   # Domain name (bbc.com)
            "snippet": r.get("snippet", ""),         # Search preview text
            "content": content,                      # Article text
            "queries": [q],                          # Which query found this
            "quality": quality or classify_source(r["domain"]),  # academic? vendor? etc.
            "source_lang": detect_lang(content),     # Language the page is written in
            "search_lang": search_lang,              # Language we searched in
            **PAGE_META.get(url, {}),                # Author(s), site name, date (for APA)
        }
        added += 1
    return added


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: this is what runs when you choose "Explore" in the launcher
# ════════════════════════════════════════════════════════════════════════

def run_explore(args):
    """
    Run the full exploratory research pipeline.

    'args' is an object containing the command-line arguments
    (or launcher choices), such as the API key, model name and
    topic. agent.py creates it with argparse.
    """

    # ── Setup ───────────────────────────────────────────────────────
    # Get the API key: from the --key argument, or from the environment variable
    # (which the launcher loaded from .env). "or" means: try the first one,
    # if it's empty/None, use the second one.
    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("No API key found -- run install.py to add one (or pass --key KEY)")  # Stop here: nothing can run without a key

    model = args.model               # Which AI model to use
    topic = args.topic or DEFAULT_TOPIC  # User's topic, or the default
    init_search(args.serper, args.brave)  # Set up search engines

    # Welcome banner
    print("=" * 60)
    print("MODE 1.a: EXPLORE (no opinion, just gather)")
    print_capabilities()                    # Show what's installed
    print(f"   Topic: {topic[:60]}...")      # Show first 60 chars of topic
    print("=" * 60)

    # This dictionary holds every source found.
    # The key is the URL (so we don't store duplicates).
    # The value is a dictionary with title, content, quality, language, etc.
    sources = {}

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: DEFINE KEY TERMS
    # ════════════════════════════════════════════════════════════════
    # Before searching, the key terms of the topic are looked up in
    # dictionaries and reference works (see define_terms). If "coding
    # assistant" has 3 different meanings, the research needs to know that
    # before it starts. The framing of the brief reads this file.

    print(f"\n{'─'*50}\nDEFINE\n{'─'*50}")

    defs = define_terms(topic, api_key, model)

    # Save the definitions to scratchpad/01_definitions.md
    pad("01_definitions", f"# Definitions\n\n{defs}")

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: NATIVE DECOMPOSITION (9 languages)
    # ════════════════════════════════════════════════════════════════
    # For each language, the AI writes search queries the way a native
    # speaker of that language would phrase them. They are not
    # translations of English queries.
    #
    # Translating "AI coding assistant" into Japanese gives the English
    # concept in Japanese characters. Japanese developers talking among
    # themselves use different terms, framing and concerns (e.g.,
    # engineer shortage instead of job loss).
    #
    # LANGUAGES.items() gives us pairs: ("en", {"name":"English",...}),
    # ("fr", {"name":"French",...}), etc.

    print(f"\n{'─'*50}\nDECOMPOSE ({len(LANGUAGES)} languages)\n{'─'*50}")

    all_queries = {}  # Will hold: {"en": [query1, query2, ...], "fr": [...], ...}

    # The topic in each language. A plain translation, kept in the
    # decomposition file so it can be checked.
    topics = {"en": topic}
    for lc, info in LANGUAGES.items():
        if lc != "en":
            t = call_ai(f"Translate this research topic into {info['name']}. Keep product and "
                        f"company names as they are. Output only the translation.\n\n{topic}",
                        api_key, model, temp=0).strip()
            topics[lc] = t if t and (lc not in NATIVE_SCRIPT or re.search(NATIVE_SCRIPT[lc], t)) else topic

    for lc, info in LANGUAGES.items():
        # lc = language code ("en", "fr", "ja", etc.)
        # info = {"name": "English", "why": "Primary global tech language"}

        # Equal query count: every language gets the same number of
        # queries (QUERIES_PER_LANGUAGE, see the top of this file).
        #
        # A much older version gave English 8 queries and other languages 3,
        # which made the corpus English-heavy by design. English still ends
        # up with more sources (more English content exists on the web), but
        # each language gets the same number of searches.
        n_queries = QUERIES_PER_LANGUAGE

        # The prompt is written in the language itself (QUERY_PROMPTS), the
        # same neutral request each time, and the topic is translated first.
        # A prompt in English that asks for "Japanese queries" gets English
        # thinking put into Japanese words.
        prompt = QUERY_PROMPTS[lc].format(n=n_queries, topic=topics[lc])

        # Send the prompt to the AI and get back the response, then parse
        # it into clean queries (see _parse_queries above for the rules).
        resp = call_ai(prompt, api_key, model, temp=0.5)
        qs = _parse_queries(resp, n_queries, lc)

        # If nothing is usable (e.g. the model dropped the native script),
        # try once more. A second sample usually comes back fine.
        if not qs:
            resp = call_ai(prompt, api_key, model, temp=0.7)
            qs = _parse_queries(resp, n_queries, lc)

        all_queries[lc] = qs  # Store the queries for this language
        print(f"   {info['name']}: {len(qs)} queries")

    # Save the decomposition to the scratchpad
    doc = "# Native Decomposition\n\n"
    for lc, qs in all_queries.items():
        doc += f"Topic as given to the {LANGUAGES[lc]['name']} prompt: {topics[lc]}\n\n" if lc != "en" else ""
        # f"## {X}\n" creates a Markdown heading
        # "\n".join(f"- {q}" for q in qs) turns the list into bullet points
        doc += f"## {LANGUAGES[lc]['name']}\n" + "\n".join(f"- {q}" for q in qs) + "\n\n"
    pad("02_decomposition", doc)

    # ════════════════════════════════════════════════════════════════
    # PHASE 3: WEB SEARCH SWEEP
    # ════════════════════════════════════════════════════════════════
    # For every query in every language, search the web and read the
    # pages found. This phase gathers most of the sources.

    print(f"\n{'─'*50}\nSEARCH\n{'─'*50}")
    print(f"   {sum(len(q) for q in all_queries.values())} queries x {RESULTS_PER_QUERY} results, "
          f"reading pages in parallel...")

    # Every query in every language, favoring results from that
    # language's region (see LANGUAGES in core.py)
    _gather([(q, RESULTS_PER_QUERY, LANGUAGES[lc]["region"], lc)
             for lc, queries in all_queries.items() for q in queries], sources)

    # How many readable pages each language's searches produced
    for lc in all_queries:
        n = sum(1 for s in sources.values() if s["search_lang"] == lc)
        print(f"   {LANGUAGES[lc]['name']:16s} {n} sources")

    # ════════════════════════════════════════════════════════════════
    # PHASE 4: AUTHORITATIVE + ACADEMIC SOURCES
    # ════════════════════════════════════════════════════════════════
    # Phase 3 searches broadly. Phase 4 searches for high-authority
    # sources: major surveys, institutional reports, and academic papers.
    # These carry more weight in a paper.
    #
    # The Japanese (AIコーディング...) and Chinese (中国AI编程...) queries
    # look for Asian-language institutional reports as well as English ones.

    print(f"\n{'─'*50}\n AUTHORITATIVE\n{'─'*50}")

    # Targeted web searches for known high-quality sources
    # (query, search region); see LANGUAGES in core.py for the regions
    added = _gather([(q, 3, region, "mixed") for q, region in [
        ("Stack Overflow developer survey AI tools 2025 2026", "us-en"),
        ("JetBrains developer ecosystem survey AI", "us-en"),
        ("OECD AI software workers report", "us-en"),
        ("EU AI Act software development", "us-en"),
        ("GitHub Copilot productivity Microsoft study", "us-en"),
        ("AIコーディングアシスタント 調査 日本", "jp-jp"),  # Japanese: AI coding assistant survey Japan
        ("中国AI编程工具 开发者 报告", "cn-zh"),            # Chinese: China AI programming tools developer report
    ]], sources)
    print(f"   {added} authoritative sources")

    # Academic paper search (Semantic Scholar + arXiv)
    # These are separate from web search. They query dedicated
    # academic databases directly through their APIs.
    for q in ["AI code generation productivity", "GitHub Copilot study", "LLM code quality"]:
        for p in search_academic(q, n=3):
            if p["url"] not in sources:
                sources[p["url"]] = {
                    "title": p["title"],
                    "domain": p["domain"],
                    "snippet": p["snippet"],
                    "content": p["snippet"],  # For papers, the snippet is the abstract
                    "queries": [q],
                    "quality": "academic",     # Papers are always academic quality
                    "year": p.get("year"),     # Publication year (for APA references)
                    "authors": p.get("authors") or [],   # Paper authors (for APA references)
                    "source_lang": "en",
                    "search_lang": "en",
                }

    # ════════════════════════════════════════════════════════════════
    # PHASE 5: EXPERIENTIAL SOURCES (developer voices)
    # ════════════════════════════════════════════════════════════════
    # Academic papers report what researchers found in controlled
    # settings. This phase collects what developers say to each other:
    # forum discussions, blog posts, personal experiences. These are
    # less rigorous but show how developers feel in day-to-day work.

    print(f"\n{'─'*50}\n EXPERIENTIAL\n{'─'*50}")

    # Search Hacker News (developer discussion forum, open API)
    for q in ["Copilot experience", "Cursor AI review", "coding assistant"]:
        for r in search_hn(q, n=5):
            if r["url"] not in sources:
                sources[r["url"]] = {
                    "title": r["title"],
                    "domain": r["domain"],
                    "snippet": r.get("snippet", ""),
                    "content": r.get("snippet", ""),  # HN gives us snippets, not full pages
                    "queries": [q],
                    "quality": "experiential/forum",
                    "source_lang": "en",
                    "search_lang": "en",
                }

    # Search for blog posts and non-English developer experiences
    added = _gather([(q, 5, region, "mixed") for q, region in [
        ("site:dev.to Copilot review developer", "us-en"),        # Dev.to (developer blog platform)
        ('"mon expérience avec Copilot" développeur', "fr-fr"),   # French: "my experience with Copilot"
        ("Copilot 使ってみた エンジニア", "jp-jp"),                 # Japanese: "tried Copilot, engineer"
    ]], sources, quality="experiential/blog")
    print(f"   {added} developer blog posts")

    # ════════════════════════════════════════════════════════════════
    # PHASE 6: RED TEAM (find weaknesses in our research)
    # ════════════════════════════════════════════════════════════════
    # The red team is an AI asked to criticize the corpus. It gets a
    # summary of the sources collected and lists what is wrong, missing
    # or unreliable, so gaps in the search show up before analysis.

    print(f"\n{'─'*50}\nRED TEAM\n{'─'*50}")

    # Count sources by language and quality tier (for the red team prompt)
    # defaultdict(int) creates a counter: lc["en"] += 1 works even if
    # lc["en"] didn't exist yet (it auto-initializes to 0).
    lc = defaultdict(int)  # Language counts
    qc = defaultdict(int)  # Quality counts

    # .values() gives only the source data (not the URLs)
    for s in sources.values():
        lc[s.get("source_lang", "?")] += 1   # Count by language
        qc[s["quality"]] += 1                 # Count by quality

    # Build a summary of all sources for the AI to review
    # list(sources.values())[:25] = first 25 sources only (to fit in the prompt)
    critique = call_ai(
        f"Red-team this research. Look for: hallucination risks, SEO sources, "
        f"contradictions, gaps, single-source claims, stale data, vendor bias.\n\n"
        f"SOURCES: {len(sources)} | LANGS: {dict(lc)} | QUALITY: {dict(qc)}\n\n"
        # Join source titles into one string, one per line
        + "\n".join(
            f"[{s['quality']}][{s.get('source_lang', '?')}] {s['title']}"
            for s in list(sources.values())[:25]
        )
        + "\n\nFor each gap, suggest a specific fix query. " + QUERY_FORMAT,
        api_key, model, temp=0.4,
        max_tokens=3000,   # Stops a model stuck repeating itself
    )

    pad("06_red_team", f"# Red Team\n\n{critique}")

    # Extract any search queries the red team suggested for fixing gaps.
    # extract_search_queries reads the "QUERY: ..." lines we asked for
    # (falling back to quoted strings if the model ignored the format).
    fixes = extract_search_queries(critique)

    # ════════════════════════════════════════════════════════════════
    # PHASE 6.5: BIAS CHECK
    # ════════════════════════════════════════════════════════════════
    # Separate from the red team because bias is a different concern.
    # The red team checks source quality and gaps.
    # The bias check looks for systematic skew in the corpus, such as:
    #   - US-centric (most sources in English from US domains)
    #   - too optimistic (mostly vendor-published "AI is great" pieces)
    #   - missing groups of people (only covers senior devs in big companies)

    print(f"\n{'─'*50}\n BIAS CHECK\n{'─'*50}")

    bias = call_ai(
        f"Bias audit. Check: geographic, optimism/pessimism, corporate, "
        f"demographic, temporal, linguistic.\nLANGS: {dict(lc)}\n"
        f"Suggest up to 12 fix queries (include non-English native script). " + QUERY_FORMAT,
        api_key, model, temp=0.4,
        max_tokens=3000,   # A real audit is ~1,000 tokens; a model once looped to 70,000
    )

    pad("065_bias", f"# Bias Audit\n\n{bias}")

    # Add the bias check's suggested fix queries to our list
    # += adds to the existing list (doesn't replace it)
    fixes += [q for q in extract_search_queries(bias) if q not in fixes]

    # ════════════════════════════════════════════════════════════════
    # PHASE 7: GAP FILL
    # ════════════════════════════════════════════════════════════════
    # Run the search queries suggested by the red team and bias check
    # to fill the gaps they identified (up to GAP_QUERIES of them).
    # Each query searches the region matching its script, so a Korean
    # fix query looks for Korean pages.

    print(f"\n{'─'*50}\nGAP FILL ({len(fixes[:GAP_QUERIES])} queries)\n{'─'*50}")

    added = _gather([(q, 3, region_for(q), "mixed") for q in fixes[:GAP_QUERIES]], sources)
    print(f"   {added} sources added")

    # ════════════════════════════════════════════════════════════════
    # SAVE: write everything to the scratchpad for downstream modes
    # ════════════════════════════════════════════════════════════════
    # This saves two files:
    #   1. corpus.json: all sources with their content (for analyze mode)
    #   2. sources_meta.json: all sources without content (for reference)
    #
    # BACKUP: Before overwriting, the existing corpus is copied to a
    # timestamped backup file. If explore runs twice and the second run
    # fails halfway, the first run's 80 good sources are preserved.
    # Backup format: corpus-YYYY.MM.DD.HH.MM.SS.json (seconds included so
    # two runs in the same minute don't overwrite each other's backup)
    #
    # This also stops two teammates from accidentally overwriting
    # each other's work if they run modes at the same time.

    # Nothing found (no internet, or every search engine blocked us): stop
    # with an error rather than replace a good corpus with an empty one
    if not sources:
        sys.exit("No sources found -- check the internet connection and try again. "
                 "Any previous corpus was kept.")

    import shutil  # For copying files (backup)
    from datetime import datetime  # Timestamps for backup filenames

    corpus_path = os.path.join(SCRATCHPAD, "corpus.json")
    meta_path = os.path.join(SCRATCHPAD, "sources_meta.json")

    # Backup existing files before overwriting
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")  # e.g., "2026.10.07.14.30.05"
    for existing_file in [corpus_path, meta_path]:
        if os.path.exists(existing_file):
            # Build backup name: corpus.json → corpus-2026.10.07.14.30.05.json
            name, ext = os.path.splitext(existing_file)
            backup = f"{name}-{timestamp}{ext}"
            shutil.copy2(existing_file, backup)  # copy2 preserves timestamps
            print(f"   Backed up: {os.path.basename(backup)}")

    # Build the corpus list. This is a "list comprehension": a compact
    # way to build a list by transforming another collection.
    #
    # enumerate(sources.items()) gives: (0, (url1, data1)), (1, (url2, data2)), ...
    # i = the index (0, 1, 2, ...), url = the URL, s = the source data
    corpus = [
        {
            "idx": i + 1,                          # Source number (1-based)
            "title": s["title"],
            "domain": s["domain"],
            "quality": s["quality"],
            "lang": s.get("source_lang", "?"),
            "url": url,
            "year": s.get("year"),                 # Known for academic papers only
            "date": s.get("date", ""),             # Publication date the page states
            "authors": s.get("authors") or [],     # Author(s) the page or paper names
            "site": s.get("site", ""),             # Site name the page states
            "content": s["content"][:CONTENT_CHARS],  # Cut long pages to CONTENT_CHARS
        }
        for i, (url, s) in enumerate(sources.items())
    ]

    # Save the corpus as JSON
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    # Save metadata (everything except the bulky content text)
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
