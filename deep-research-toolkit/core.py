"""core.py: shared functions that every mode imports.
How it works:
  Search: web (ddgs, Google fallback), papers (Semantic Scholar, arXiv), Hacker News
  Extraction: reads article text with trafilatura, camoufox for blocked sites
  AI: sends prompts to models through the OpenRouter API
  Scratchpad: saves and loads each mode's work
  Sources: classifies sources by type, writes APA references, detects language
  Evidence cards: one short English summary per source, cached"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS AND OPTIONAL LIBRARIES: each optional one sets a flag
# ════════════════════════════════════════════════════════════════════════

import json
import re
import sys
import os
import importlib.util
import requests
from urllib.parse import urlparse
from pathlib import Path

# ddgs queries several engines; duckduckgo_search is its old name (now returns nothing)
try:
    from ddgs import DDGS
    _SEARCH_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _SEARCH_DDG = True
    except ImportError:
        _SEARCH_DDG = False

# Google fallback: installs as multi_search_engine, imports as multi_search
try:
    from multi_search import search as _google_search
    _SEARCH_GOOGLE = True
except ImportError:
    _SEARCH_GOOGLE = False

if _SEARCH_DDG or _SEARCH_GOOGLE:
    _SEARCH = "available"
else:
    _SEARCH = "none"

try:
    import trafilatura
    from trafilatura.settings import use_config as _traf_config
    _TRAF = True
except ImportError:
    _TRAF = False

# 15 s page timeout; silence per-page download errors
if _TRAF:
    _TRAF_CONFIG = _traf_config()
    _TRAF_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "15")
    import logging
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

# Camoufox: stealth Firefox for bot-protected sites
try:
    from camoufox.sync_api import Camoufox
    from camoufox.pkgman import installed_verstr
    installed_verstr()   # Raises if the ~300MB browser is not downloaded
    _CAMO = True
except Exception:
    _CAMO = False

try:
    from pypdf import PdfReader   # Reads PDF sources (reports, papers)
    _PDF = True
except ImportError:
    _PDF = False

try:
    from langdetect import detect as _detect_lang
    _LANG = True
except ImportError:
    _LANG = False
    def _detect_lang(t):
        return "unknown"

try:
    from semanticscholar import SemanticScholar
    _S2 = True
except ImportError:
    _S2 = False

try:
    import arxiv as _arxiv
    _ARXIV = True
except ImportError:
    _ARXIV = False

try:
    from hn import search_by_date as _hn_search
    _HN = True
except ImportError:
    _HN = False

_CITE = importlib.util.find_spec("citationlib") is not None   # Checked without importing it


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

SCRATCHPAD = "scratchpad"  # Folder every mode reads and writes

# Search languages: "region" favors that country's results, "why" goes into explore's query prompt
LANGUAGES = {
    "en": {"name": "English",         "region": "us-en", "why": "Primary global tech language"},
    "fr": {"name": "French",          "region": "fr-fr", "why": "EU AI Act, Sciences Po context"},
    "de": {"name": "German",          "region": "de-de", "why": "Engineering culture, privacy-first, EU voice"},
    "ja": {"name": "Japanese",        "region": "jp-jp", "why": "Corporate dev culture, engineer shortage framing"},
    "ko": {"name": "Korean",          "region": "kr-kr", "why": "Samsung/Naver, own AI tools"},
    "zh": {"name": "Chinese",         "region": "cn-zh", "why": "Different AI ecosystem (Tongyi Lingma, CodeGeeX)"},
    "pt": {"name": "Portuguese (BR)", "region": "br-pt", "why": "Brazil #2 on GitHub"},
    "es": {"name": "Spanish",         "region": "xl-es", "why": "LatAm dev context"},  # xl = Latin America
    "ru": {"name": "Russian",         "region": "ru-ru", "why": "JetBrains origin, strong dev tradition"},
}

# LLM filler phrases the voice cleaner in synthesize.py removes
BANNED_PHRASES = [
    "it's important to note", "it is important to note",
    "it's worth noting", "it is worth noting",
    "in today's rapidly evolving", "in an era of",
    "in the rapidly evolving landscape",
    "paradigm shift", "game-changer", "game changer",
    "revolutionize", "revolutionizing", "transformative potential",
    "unprecedented", "cutting-edge", "groundbreaking",
    "deep dive", "delve", "delving into",
    "holistic approach", "robust framework",
    "navigate the complexities", "double-edged sword",
    "the landscape of", "at the forefront",
    "and that's okay", "embrace the change", "lean into",
    "furthermore", "moreover", "additionally",
    "in conclusion", "to summarize",
    "as we have seen", "as discussed above",
    "tip of the iceberg", "opens the door",
    "paves the way", "silver bullet",
    "low-hanging fruit", "move the needle",
    "reshaping the landscape", "spearheading",
    "synergy", "leverage the power",

    # French, German, Japanese, Spanish, Portuguese
    "il est important de noter", "il convient de souligner",
    "dans un monde en constante évolution", "force est de constater",
    "en effet", "par ailleurs", "en conclusion",
    "un véritable changement de paradigme",
    "es ist wichtig zu beachten", "hervorzuheben ist",
    "in der heutigen schnelllebigen", "zusammenfassend lässt sich sagen",
    "darüber hinaus", "des Weiteren",
    "重要なのは", "注目すべきは",
    "言うまでもなく",
    "まとめると",
    "es importante destacar", "cabe señalar",
    "en un mundo en constante evolución", "en conclusión",
    "é importante notar", "vale ressaltar",
    "em um mundo em constante evolução", "em conclusão",
]


# ════════════════════════════════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════════════════════════════════

def init_search(serper_key=None, brave_key=None):
    """No-op: ddgs needs no setup; the keys are unused."""
    pass


# Searches that found nothing, and why, for the summary search_many prints
_SEARCH_FAILS = []


def search_web(query, n=5, region="us-en"):
    """Return up to n results as {title, url, snippet, domain}: ddgs first, Google if it finds nothing."""
    import time
    results, reason = [], ""

    # Tier 1: ddgs, three tries with a pause (engines slow down rapid requests)
    for attempt in range(3):
        if not _SEARCH_DDG or results:
            break
        if attempt:
            time.sleep(3 * attempt)
        try:
            for r in DDGS().text(query, region=region, max_results=n):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", ""),
                })
        except Exception as e:
            reason = "timed out" if "timeout" in f"{type(e).__name__} {e}".lower() else "no results"

    # Tier 2: Google; results come as dicts or (title, url[, snippet]) tuples
    if not results and _SEARCH_GOOGLE:
        try:
            raw = _google_search(query, num_results=n, search_engine="google")

            for r in (raw or [])[:n]:
                if isinstance(r, dict):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", r.get("url", r.get("href", ""))),
                        "snippet": r.get("snippet", r.get("description", "")),
                    })
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    results.append({
                        "title": str(r[0]),
                        "url": str(r[1]),
                        "snippet": str(r[2]) if len(r) > 2 else "",
                    })
        except Exception:
            pass   # Counted below as a search that found nothing

    if not results:
        _SEARCH_FAILS.append(reason or "no results")

    for r in results:
        r["domain"] = urlparse(r.get("url", "")).netloc

    return [r for r in results if r["domain"] and r["url"].startswith("http")]


def search_academic(query, n=5):
    """Return papers from Semantic Scholar and arXiv as {title, url, snippet, domain, year, authors, citations}."""
    papers = []

    if _S2:
        try:
            # retry=False: without a key S2 rate-limits and its retries wait minutes
            for p in SemanticScholar(retry=False, timeout=20).search_paper(query, limit=n):
                papers.append({
                    "title": p.title or "",
                    "domain": "semanticscholar.org",
                    "url": p.url or f"https://www.semanticscholar.org/paper/{p.paperId}",
                    "snippet": p.abstract or "",
                    "year": getattr(p, "year", None),
                    "authors": [a.name for a in (getattr(p, "authors", None) or []) if getattr(a, "name", None)],
                    "citations": getattr(p, "citationCount", None),
                })
        except Exception as e:
            print(f"      Semantic Scholar error: {type(e).__name__}: {e}")

    if _ARXIV:
        try:
            client = _arxiv.Client()
            search = _arxiv.Search(query=query, max_results=n,
                                   sort_by=_arxiv.SortCriterion.Relevance)
            for p in client.results(search):
                papers.append({
                    "title": p.title,
                    "domain": "arxiv.org",
                    "url": p.entry_id,
                    "snippet": p.summary or "",
                    "year": p.published.year if p.published else None,
                    "authors": [a.name for a in (p.authors or [])],
                })
        except Exception as e:
            print(f"      arXiv error: {type(e).__name__}: {e}")

    return papers


def search_hn(query, n=10):
    """Return up to n Hacker News stories since 2024 as {title, url, snippet, domain, points}."""
    results = []

    if _HN:
        try:
            for item in _hn_search(query, stories=True, created_at__gt="2024-01-01"):
                if len(results) >= n:
                    break

                results.append({
                    "title": item.get("title") or "",
                    "url": f"https://news.ycombinator.com/item?id={item.get('objectID', '')}",
                    "snippet": item.get("story_text") or "",
                    "domain": "news.ycombinator.com",
                    "points": item.get("points", 0),
                })
        except Exception as e:
            print(f"      HN search error: {type(e).__name__}: {e}")
    else:
        results = search_web(f"site:news.ycombinator.com {query}", n=n)

    return results


# ════════════════════════════════════════════════════════════════════════
# CONTENT EXTRACTION
# ════════════════════════════════════════════════════════════════════════

CONTENT_CHARS = 30000  # Characters kept per source (keeps 93% of measured page text)


# Search ads and redirect links (sponsored pages, not sources) and video pages
_AD_URL = re.compile(r"bing\.com/(aclick|ck/a)|googleadservices\.|doubleclick\.net|google\.[a-z.]+/aclk|duckduckgo\.com/y\.js"
                     r"|youtube\.com/|youtu\.be/", re.I)   # Video pages: no article text to read

# Error, block and bot-check pages, recognised by their opening words
_ERROR_PAGE = re.compile(
    r"are you a robot|captcha|verify you are (a )?human|security verification|performing security|"
    r"access denied|403 forbidden|http status 404|page not found|404 not found|something.s wrong here|"
    r"enable javascript|just a moment\.\.\.|checking your browser", re.I)


def page_problem(text):
    """Return why a page is not a usable source ("blocked or error page", ...), or "" if it is."""
    t = (text or "").strip()
    if len(t) < 200:
        return "too short or empty"
    sample = t[:3000]
    readable = sum(ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()[]-%/&" for ch in sample)
    if readable / len(sample) < 0.75 or sample.count("�") > 5:
        return "unreadable (binary or garbled)"
    if _ERROR_PAGE.search(t[:400]):
        return "blocked or error page"
    if t[:300].count("{{") >= 2:
        return "page template, no content"
    return ""


def _read_pdf(url, max_chars):
    """Return the text of a PDF (with pypdf), or None; arXiv /pdf/ links are PDFs too."""
    if not _PDF:
        return None
    import io
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return None
    reader = PdfReader(io.BytesIO(r.content))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:max_chars] if len(text) > 100 else None


PAGE_META = {}  # url → {authors, site, date}, for APA references


def _page_meta(html, url=None):
    """Return {"authors": [...], "site": "...", "date": "YYYY-MM-DD"} from a page's metadata tags."""
    try:
        doc = trafilatura.extract_metadata(html, default_url=url,
                                           date_config={"original_date": True, "extensive_search": True})
    except Exception:
        doc = None
    if not doc:
        return {"authors": [], "site": "", "date": ""}
    authors = [a.strip() for a in re.split(r";|\band\b|\s&\s", doc.author or "") if a.strip()]
    date = doc.date or ""
    # A /YYYY/MM/DD/ date in the URL wins: the tags sometimes give the edit date
    m = re.search(r"/(20\d\d)/(\d{1,2})/(\d{1,2})/", url or "")
    if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
        date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return {"authors": authors[:20], "site": (doc.sitename or "").strip(),
            "date": date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else ""}


def fetch_page(url, max_chars=CONTENT_CHARS, use_browser=True):
    """Return a page's article text (up to max_chars) or None: trafilatura, then camoufox if use_browser."""

    # PDFs: read with pypdf (trafilatura would return the raw bytes as text)
    if re.search(r"\.pdf($|\?)|arxiv\.org/pdf/", url, re.I):
        try:
            return _read_pdf(url, max_chars)
        except Exception:
            return None

    # Tier 1: trafilatura; under 100 characters counts as a failed extraction
    if _TRAF:
        try:
            html = trafilatura.fetch_url(url, config=_TRAF_CONFIG)

            if html:
                text = trafilatura.extract(html, include_tables=True, favor_recall=True)

                if text and len(text.strip()) > 100:
                    PAGE_META[url] = _page_meta(html, url)
                    return text[:max_chars]
        except Exception:
            pass

    # Tier 2: camoufox stealth browser for bot-protected sites
    if _CAMO and use_browser:
        try:
            with Camoufox(headless=True) as browser:
                return _read_with_browser(browser, url, max_chars)
        except Exception as e:
            print(f"      camoufox extraction error: {type(e).__name__}")

    return None


# Browser-side JS: drop nav, header, footer and sidebars, return the main text
_EXTRACT_JS = """() => {
    for (const s of ['nav','header','footer','aside','.sidebar'])
        document.querySelectorAll(s).forEach(e => e.remove());
    const m = document.querySelector('main,article,[role="main"]');
    return (m || document.body).innerText;
}"""


def _read_with_browser(browser, url, max_chars):
    """Open url in an already-running camoufox browser; return its text, or None."""
    page = browser.new_page()
    try:
        page.goto(url, timeout=20000)
        page.wait_for_load_state("domcontentloaded")
        text = page.evaluate(_EXTRACT_JS)
        if text and len(text.strip()) > 100:
            return re.sub(r"\s+", " ", text).strip()[:max_chars]
        return None
    finally:
        page.close()


def fetch_pages(urls, max_chars=CONTENT_CHARS, workers=8):
    """Return {url: text} for the pages read: trafilatura in parallel, then camoufox retries one by one."""
    # No page reader: stop instead of a long run that saves only snippets
    if urls and not (_TRAF or _CAMO):
        sys.exit("\n   No page reader is installed (trafilatura). "
                 "Run install.py, or: pip install trafilatura")

    import time
    started = time.time()
    urls = list(dict.fromkeys(urls))
    ads = [u for u in urls if _AD_URL.search(u)]
    urls = [u for u in urls if not _AD_URL.search(u)]
    print(f"      reading {len(urls)} pages ({workers} at a time)...")
    texts = _read_pages_isolated(urls, max_chars, workers)
    pages = {u: texts[u] for u in urls if texts.get(u)}

    # Retry failures in one shared browser
    failed = [u for u in urls if u not in pages]
    if _CAMO and failed:
        print(f"      retrying {len(failed)} blocked pages in the stealth browser...")
        try:
            with Camoufox(headless=True) as browser:
                for u in failed:
                    try:
                        text = _read_with_browser(browser, u, max_chars)
                        if text:
                            pages[u] = text
                    except Exception:
                        continue
        except Exception as e:
            print(f"      camoufox unavailable: {type(e).__name__}")

    # Leave out error pages, bot checks, binary text and near-empty pages
    from collections import Counter
    dropped = Counter()
    for u in list(pages):
        why = page_problem(pages[u])
        if why:
            dropped[why] += 1
            del pages[u]
    if ads:
        dropped["search ad or video"] = len(ads)
    print(f"      read {len(pages)} of {len(urls)} pages in {time.time() - started:.0f}s"
          + (f"; left out: {', '.join(f'{n} {why}' for why, n in dropped.items())}" if dropped else ""))
    return pages


def _read_pages_isolated(urls, max_chars, workers):
    """Return {url: text or None}, read by trafilatura `workers` at a time in a helper process.
    Helper process: a C-library crash (lxml "double free", segfault) kills only the helper."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    code = ("import faulthandler, sys; faulthandler.enable(); "
            f"sys.path.insert(0, {here!r}); import core; core._read_pages_worker()")

    def run_helper(todo, n_workers):
        """Read `todo` in one helper; returns (pages being read at exit, exit code)."""
        proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, encoding="utf-8")
        proc.stdin.write(json.dumps({"urls": todo, "max_chars": max_chars, "workers": n_workers}))
        proc.stdin.close()
        reading = set()
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "start" in msg:
                reading.add(msg["start"])
            elif "done" in msg:
                reading.discard(msg["done"])
                results[msg["done"]] = msg["text"]
                if msg.get("meta"):
                    PAGE_META[msg["done"]] = msg["meta"]
        return reading, proc.wait()

    results = {}
    jobs = [(list(urls), workers)]
    crashes = 0
    while jobs and crashes < 10:
        todo, n_workers = jobs.pop(0)
        todo = [u for u in todo if u not in results]
        if not todo:
            continue
        reading, exit_code = run_helper(todo, n_workers)
        if exit_code == 0:
            continue
        # Crash: skip the page if read alone, else re-read the in-flight pages one at a time
        crashes += 1
        rest = [u for u in todo if u not in results and u not in reading]
        if n_workers == 1 and reading:
            results.update(dict.fromkeys(reading))
            print(f"      skipped a page that crashes the page reader: {next(iter(reading))[:80]}",
                  flush=True)
        elif reading:
            print(f"      the page reader crashed (exit code {exit_code}) -- re-reading the "
                  f"{len(reading)} pages it was on one at a time, then the other {len(rest)}",
                  flush=True)
            jobs.insert(0, (sorted(reading), 1))
        else:
            print(f"      the page reader won't start (exit code {exit_code})", flush=True)
            break
        if rest:
            jobs.insert(1 if reading and n_workers > 1 else 0, (rest, n_workers))
    return results


def _read_pages_worker():
    """Helper process: read the JSON job from stdin, print a "start" and a "done" JSON line per page."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    job = json.loads(sys.stdin.read())
    lock = threading.Lock()

    def emit(msg):
        with lock:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    def read_one(url):
        emit({"start": url})
        text = fetch_page(url, job["max_chars"], use_browser=False)
        emit({"done": url, "text": text, "meta": PAGE_META.get(url)})

    with ThreadPoolExecutor(max_workers=job["workers"]) as pool:
        list(pool.map(read_one, job["urls"]))


def region_for(query):
    """Return the search region for a query's script (Hangul, Kana, Han, Cyrillic), else us-en."""
    for lang, script in [("ko", r"[\uac00-\ud7a3]"),
                         ("ja", r"[\u3040-\u30ff]"),     # Kana before Han: Japanese also uses Kanji
                         ("zh", r"[\u4e00-\u9fff]"),
                         ("ru", r"[\u0400-\u04ff]")]:
        if re.search(script, query):
            return LANGUAGES[lang]["region"]
    return "us-en"


def search_many(jobs, workers=3):
    """Run (query, n, region) searches a few at a time; return the result lists in order."""
    from concurrent.futures import ThreadPoolExecutor

    # No search library: stop instead of reporting 0 results
    if jobs and _SEARCH == "none":
        sys.exit("\n   No web search library is installed (ddgs). "
                 "Run install.py, or: pip install ddgs")
    _SEARCH_FAILS.clear()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(lambda job: search_web(job[0], n=job[1], region=job[2]), jobs))
    # One summary line instead of an error per search
    if _SEARCH_FAILS:
        timed_out = _SEARCH_FAILS.count("timed out")
        print(f"      {len(_SEARCH_FAILS)} of {len(jobs)} searches found nothing"
              + (f" ({timed_out} timed out after 3 tries)" if timed_out else "")
              + "; the run continues with the others")
    return found


# ════════════════════════════════════════════════════════════════════════
# AI: OpenRouter API
# ════════════════════════════════════════════════════════════════════════

def call_ai(prompt, api_key, model, temp=0.6, system="", max_tokens=16000):
    """Send a prompt to a model through OpenRouter; return the reply text, or "" on failure.
    max_tokens caps the reply so a model stuck in a loop stops."""

    msgs = []

    if system:
        msgs.append({"role": "system", "content": system})

    msgs.append({"role": "user", "content": prompt})

    import time

    # Retry 429, 5xx and dropped connections after 5, 20, 60 s; other errors fail at once
    waits = [5, 20, 60]
    for attempt in range(len(waits) + 1):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": msgs,
                    "temperature": temp,
                    **({"max_tokens": max_tokens} if max_tokens else {}),
                },
                timeout=180,
            )

            temporary = r.status_code == 429 or r.status_code >= 500
            if temporary and attempt < len(waits):
                wait = waits[attempt]
                if r.headers.get("Retry-After", "").isdigit():
                    wait = min(int(r.headers["Retry-After"]), 120)
                print(f"   AI busy ({r.status_code}) -- retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue

            r.raise_for_status()

            choice = r.json()["choices"][0]
            # A reply the provider broke off ("error") is retried; "length" (max_tokens
            # reached) and "stop" are complete replies
            if choice.get("finish_reason") == "error" and attempt < len(waits):
                print(f"   AI reply broke off -- retrying in {waits[attempt]}s", flush=True)
                time.sleep(waits[attempt])
                continue
            return choice["message"]["content"] or ""

        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < len(waits):
                print(f"   AI connection problem ({type(e).__name__}) -- retrying in "
                      f"{waits[attempt]}s", flush=True)
                time.sleep(waits[attempt])
                continue
            print(f"   AI error: {e}")
            return ""
        except Exception as e:
            print(f"   AI error: {e}")
            return ""
    return ""


# ════════════════════════════════════════════════════════════════════════
# SCRATCHPAD: each mode's saved work
# ════════════════════════════════════════════════════════════════════════

def pad(name, content):
    """Save content to scratchpad/<name>.md."""
    Path(SCRATCHPAD).mkdir(exist_ok=True)

    p = os.path.join(SCRATCHPAD, f"{name}.md")

    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"   {p}")


def load_pad(name):
    """Return scratchpad/<name>.md, or "" if it does not exist."""
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return ""


def load_sources():
    """Return the newest research_*_meta.json (explore's source inventory), or None."""
    for fn in sorted(Path(".").glob("research_*_meta.json"), reverse=True):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)

    return None


def load_corpus_texts():
    """Return every scratchpad .md file as {"file": name, "text": contents}."""
    texts = []
    for fn in sorted(Path(SCRATCHPAD).glob("*.md")):
        with open(fn, encoding="utf-8") as f:
            texts.append({"file": fn.name, "text": f.read()})
    return texts


# ════════════════════════════════════════════════════════════════════════
# SOURCE CLASSIFICATION AND APA REFERENCES
# ════════════════════════════════════════════════════════════════════════

# Plain names of the classify_source types, shown in charts and the brief
SOURCE_LABELS = {
    "academic": "Research papers",
    "intl institution": "International bodies (OECD, UN, EU)",
    "government": "Government",
    "consulting": "Consulting firms",
    "vendor/primary": "AI tool makers (GitHub, OpenAI...)",
    "experiential/forum": "Developer forums",
    "experiential/blog": "Developer blogs",
    "journalism": "News media",
    "SEO risk": "Low-quality content sites",
    "unclassified": "Other websites",
}


def source_label(kind):
    """The plain name of a source type: "vendor/primary" → "AI tool makers (GitHub, OpenAI...)"."""
    return SOURCE_LABELS.get(kind or "unclassified", kind or "Other websites")


def classify_source(domain):
    """Return a source type ("academic", "journalism", "vendor/primary"...) from a domain name."""
    d = domain.lower()

    def ends(*suffixes):
        return d.endswith(suffixes)

    if (any(x in d for x in ["arxiv", "ieee", "acm.org", "nature.com", "semanticscholar",
                             "ssrn.com", "researchgate", "sciencedirect", "springer"])
            or ends(".edu", ".ac.jp", ".ac.kr", ".ac.uk", ".edu.cn")):
        return "academic"

    if any(x in d for x in ["oecd", "ilo.org", "weforum", "europa.eu", "worldbank"]):
        return "intl institution"

    if ends(".gov", ".gouv.fr", ".go.jp", ".go.kr", ".gov.cn", ".gov.uk", ".gob.mx", ".gov.br"):
        return "government"

    if any(x in d for x in ["capgemini", "rolandberger", "nomura", "deloitte", "kpmg", "mckinsey",
                            "pwc", "accenture", "bcg.com", "bain.com", "gartner"]):
        return "consulting"

    # AI coding tool makers; plain github.com is mostly user repositories, so left out
    if any(x in d for x in ["github.blog", "docs.github.com", "microsoft.com", "openai.com",
                            "jetbrains", "stackoverflow.blog", "anthropic.com", "cursor.com",
                            "ibm.com", "tabnine", "codeium", "windsurf", "replit.com",
                            "sourcegraph", "kimi.ai", "aliyun.com"]):
        return "vendor/primary"

    if any(x in d for x in ["news.ycombinator", "dev.to", "lobste.rs", "reddit.com", "qiita.com",
                            "zenn.dev", "note.com", "tistory.com", "brunch.co.kr", "velog.io",
                            "habr.com"]):
        return "experiential/forum"

    if any(x in d for x in ["reuters", "ft.com", "nytimes", "bloomberg", "techcrunch", "wired",
                            "lemonde", "lesechos", "heise.de", "golem.de", "zdnet", "theverge",
                            "arstechnica", "cio.com", "infoq", "itmedia", "nikkei", "sedaily",
                            "inews24", "vc.ru"]):
        return "journalism"

    if any(x in d for x in ["medium.com", "towardsdatascience", "analyticsvidhya"]):
        return "SEO risk"

    return "unclassified"


# Site names the web address spells wrong ("stackoverflow" → "Stack Overflow")
SITE_NAMES = {"github": "GitHub", "stackoverflow": "Stack Overflow", "semanticscholar": "Semantic Scholar",
              "arxiv": "arXiv", "mckinsey": "McKinsey & Company", "ibm": "IBM", "jetbrains": "JetBrains",
              "linkedin": "LinkedIn", "youtube": "YouTube", "techcrunch": "TechCrunch", "openai": "OpenAI",
              "deepmind": "DeepMind", "nytimes": "The New York Times", "wsj": "The Wall Street Journal",
              "ft": "Financial Times", "bbc": "BBC", "oecd": "OECD", "nist": "NIST", "mit": "MIT",
              "hbr": "Harvard Business Review", "bcg": "Boston Consulting Group", "pwc": "PwC",
              "kpmg": "KPMG", "ey": "EY", "wikipedia": "Wikipedia", "medium": "Medium", "dev": "DEV Community",
              "qiita": "Qiita", "zenn": "Zenn", "naver": "Naver", "csdn": "CSDN", "habr": "Habr"}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def site_name(s):
    """Return the organization behind a source: its stated site name, else one built from the domain."""
    parts = [p for p in (s.get("domain") or "").lower().split(".") if p not in ("www", "m")]
    if len(parts) >= 3 and parts[-2] in ("co", "com", "or", "ac", "gov", "org", "ne", "go", "edu"):
        key = parts[-3]
    else:
        key = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
    # Stated name, unless it only repeats the domain ("glean.com", "Stackoverflow")
    site = (s.get("site") or "").strip()
    if site and "." not in site and site.lower().replace(" ", "") != key:
        return site
    return SITE_NAMES.get(key, " ".join(w.capitalize() for w in key.split("-")))


# Words that mark an author name as an organization ("The Glean Team")
_GROUP_WORDS = {"team", "staff", "editor", "editors", "editorial", "newsroom", "inc", "ltd", "llc",
                "gmbh", "corp", "corporation", "group", "press", "media", "news", "blog", "admin",
                "rédaction", "redaction", "redaktion", "research", "institute", "university", "foundation"}


def _is_group(name):
    """True if an author name is an organization rather than a person."""
    words = re.findall(r"[^\W\d_]+", name.lower())
    return name.lower().startswith("the ") or bool(_GROUP_WORDS & set(words)) or len(words) > 4


def _apa_person(name):
    """A person's name as APA writes it: "Jane A. Doe" → "Doe, J. A."; "Doe, Jane" → "Doe, J."."""
    if _is_group(name):
        return name
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
    else:
        words = name.split()
        if len(words) < 2:
            return name
        last, first = words[-1], " ".join(words[:-1])
    initials = " ".join(f"{w[0]}." for w in re.split(r"[\s-]+", first) if w and w[0].isalpha())
    return f"{last}, {initials}" if initials else last


def _apa_surname(name):
    """The surname of a person, for in-text citations."""
    if _is_group(name):
        return name
    return name.split(",")[0].strip() if "," in name else name.split()[-1]


def _apa_date(s, year_only=False):
    """Return "2024, March 5", the year alone, or "n.d." when undated."""
    date = s.get("date") or ""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date)
    if m and not year_only:
        return f"{m.group(1)}, {_MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}"
    return (m.group(1) if m else str(s.get("year") or "")) or "n.d."


def apa_reference(s):
    """Return a source as an APA 7 reference; with no person named, the site is the author:
    Doe, J., & Lee, K. (2024, March 5). *Title* [English title]. Site. URL"""
    title = (s.get("title") or s.get("domain") or "").strip().rstrip(".")
    translated = f" [{s['title_en'].strip().rstrip('.')}]" if s.get("title_en") else ""
    people = [_apa_person(a) for a in (s.get("authors") or [])][:20]
    date = _apa_date(s)
    url = s.get("url", "")
    if people:
        names = people[0] if len(people) == 1 else ", ".join(people[:-1]) + ", & " + people[-1]
        names += "" if names.endswith(".") else "."
        return f"{names} ({date}). *{title}*{translated}. {site_name(s)}. {url}".strip()
    return f"{site_name(s)}. ({date}). *{title}*{translated}. {url}".strip()


def apa_in_text(sources):
    """Return an APA in-text citation of sources, sorted: (Doe & Lee, 2024; GitHub, n.d.)."""
    def one(s):
        people = s.get("authors") or []
        if len(people) == 1:
            who = _apa_surname(people[0])
        elif len(people) == 2:
            who = f"{_apa_surname(people[0])} & {_apa_surname(people[1])}"
        elif people:
            who = f"{_apa_surname(people[0])} et al."
        else:
            who = site_name(s)
        return f"{who}, {_apa_date(s, year_only=True)}"
    return "(" + "; ".join(sorted(dict.fromkeys(one(s) for s in sources))) + ")"


TITLES_FILE = "title_translations.json"


def translate_titles(corpus, api_key, model, corpus_dir=SCRATCHPAD):
    """Set s["title_en"] on non-English sources (AI-translated, cached); return the cache {url: title}."""
    path = os.path.join(corpus_dir, TITLES_FILE)
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [s for s in corpus if s.get("lang") not in ("en", "unknown", "?", None, "")
            and s.get("title") and s.get("url") not in cache]
    for start in range(0, len(todo), 60):
        batch = todo[start:start + 60]
        reply = call_ai("Translate each title into English. One line each, starting with its "
                        "label:\nT1: <translation>\n\n"
                        + "\n".join(f"T{i}: {s['title']}" for i, s in enumerate(batch, 1)),
                        api_key, model, temp=0) or ""
        for m in re.finditer(r"^\W*T(\d+)\W*:\s*(.+)$", reply, re.M):
            i = int(m.group(1)) - 1
            if 0 <= i < len(batch):
                cache[batch[i]["url"]] = m.group(2).strip()
    if todo:
        Path(corpus_dir).mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, ensure_ascii=False)
    for s in corpus:
        if s.get("url") in cache:
            s["title_en"] = cache[s["url"]]
    return cache


def load_title_translations(corpus, corpus_dir=SCRATCHPAD):
    """Set s["title_en"] from the cache translate_titles saved (no AI call)."""
    path = os.path.join(corpus_dir, TITLES_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        for s in corpus:
            if s.get("url") in cache:
                s["title_en"] = cache[s["url"]]


def detect_lang(text):
    """Return the base language code ("en", "zh") of text over 50 characters, else "unknown".
    Korean, Japanese, Chinese and Russian are told by their script, which the detector gets wrong."""
    sample = (text or "")[:2000]
    letters = len(re.findall(r"[^\W\d_]", sample)) or 1
    share = {lang: len(re.findall(script, sample)) / letters
             for lang, script in [("ko", r"[\uac00-\ud7a3]"), ("kana", r"[\u3040-\u30ff]"),
                                  ("han", r"[\u4e00-\u9fff]"), ("ru", r"[\u0400-\u04ff]")]}
    if share["ko"] > 0.2:
        return "ko"
    if share["kana"] + share["han"] > 0.2:
        return "ja" if share["kana"] > 0.03 else "zh"   # Japanese mixes kana into its kanji
    if share["ru"] > 0.2:
        return "ru"
    if _LANG and text and len(text) > 50:
        try:
            return _detect_lang(text[:500]).split("-")[0]
        except Exception:
            pass
    return "unknown"


# ════════════════════════════════════════════════════════════════════════
# CAPABILITIES AND QUERY PARSING
# ════════════════════════════════════════════════════════════════════════

def print_capabilities():
    """Print which optional libraries are installed ("OK") or missing ("--")."""
    def ok(flag, missing="--"):
        return "OK" if flag else missing

    print(f"   Search:      DDG {ok(_SEARCH_DDG)} + Google fallback {ok(_SEARCH_GOOGLE)}")
    print(f"   Extract:     trafilatura {ok(_TRAF)} + camoufox {ok(_CAMO)}")
    print(f"   Academic:    S2 {ok(_S2)} + arXiv {ok(_ARXIV)}")
    print(f"   HN:          {ok(_HN, '(web fallback)')}")
    print(f"   Lang detect: {ok(_LANG)}")
    print(f"   APA:         {ok(_CITE, '(manual fallback)')}")


def extract_quoted_queries(text, min_len=10):
    """Return one-line quoted strings (straight or curly quotes) of at least min_len characters."""
    quoted = re.findall(r'["\u201c]([^"\u201c\u201d\n]*)["\u201d]', text or "")
    return [q.strip() for q in quoted if len(q.strip()) >= min_len]


# Prompt line asking the AI to mark its suggested searches
QUERY_FORMAT = ("Put each suggested search query on its own line, "
                "starting with 'QUERY: ' (no quotes around it).")


def extract_search_queries(text, min_len=10):
    """Return the "QUERY: ..." lines of an AI reply (quoted strings if none), near-duplicates dropped."""
    marked = re.findall(r"^[\s>*\-•\d.)]*\**QUERY\**\s*[:：]\s*(.+)$", text or "",
                        re.MULTILINE | re.IGNORECASE)
    candidates = [q.strip(' *`"\u201c\u201d') for q in marked] or extract_quoted_queries(text, min_len)

    queries, seen = [], set()
    for q in candidates:
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if len(q) >= min_len and key not in seen:
            seen.add(key)
            queries.append(q)
    return queries


# ════════════════════════════════════════════════════════════════════════
# BATCHING: more text than fits in one prompt
# ════════════════════════════════════════════════════════════════════════

def make_batches(parts, batch_chars=12000):
    """Pack (label, text) parts in order into batches of at most batch_chars; long parts split into labelled pieces."""
    batches = []
    current = ""

    for label, text in parts:
        room = batch_chars - len(label) - 30
        pieces = [text[i:i + room] for i in range(0, len(text), room)] or [""]

        for k, piece in enumerate(pieces, 1):
            name = label if len(pieces) == 1 else f"{label} (part {k})"
            block = f"── {name} ──\n{piece}\n"

            if current and len(current) + len(block) > batch_chars:
                batches.append(current)
                current = ""
            current += block

    if current:
        batches.append(current)
    return batches


def ai_map(batches, make_prompt, api_key, model, temp=0.2, workers=4):
    """Run make_prompt(batch) through call_ai, `workers` at a time; return answers in batch order ("" on failure)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(batch):
        return call_ai(make_prompt(batch), api_key, model, temp=temp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, batches))


# ════════════════════════════════════════════════════════════════════════
# EVIDENCE CARDS: one short English summary per source
# ════════════════════════════════════════════════════════════════════════

CARDS_FILE = "evidence_cards.json"
NO_EVIDENCE = "NO EVIDENCE"  # Card text for pages with nothing usable
CARD_VERSION = "v2"          # Change when the card prompt changes, so old cards are rewritten
CHARS_PER_CARD_SENTENCE = 3000   # One card sentence per 3,000 characters of source text
MAX_CARD_SENTENCES = 6           # Sentence cap per card


def _card_sentences(item):
    """How many sentences the card of a source may have, from its length."""
    return max(1, min(MAX_CARD_SENTENCES, 1 + len(item.get("content", "")) // CHARS_PER_CARD_SENTENCE))


def _content_key(item):
    """Fingerprint of a source's text, to notice when it changed."""
    import hashlib
    return hashlib.sha1(item.get("content", "").encode("utf-8")).hexdigest()[:12]


def build_evidence_cards(corpus, api_key, model, corpus_dir=SCRATCHPAD):
    """Return {idx: card text} for every source, using and updating the evidence_cards.json cache."""
    cache_path = os.path.join(corpus_dir, CARDS_FILE)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    # Sources with no card, changed text, or a card from an older CARD_VERSION
    def key(item):
        return f"{CARD_VERSION}:{_content_key(item)}"
    todo = [s for s in corpus if cache.get(str(s["idx"]), {}).get("key") != key(s)]

    # Two passes: the second retries sources the AI skipped
    for attempt in range(2):
        if not todo:
            break
        if attempt == 0:
            print(f"   Writing evidence cards for {len(todo)} sources...")
        parts = [(f"[{s['idx']}] ({s.get('lang', '?')}) {s.get('title', '')} -- {s.get('domain', '')}"
                  f" -- up to {_card_sentences(s)} sentence{'s' if _card_sentences(s) > 1 else ''}",
                  s.get("content", "")) for s in todo]
        answers = ai_map(
            # 80,000 characters per call, so no source (<= CONTENT_CHARS) is split
            make_batches(parts, batch_chars=80000),
            lambda batch: (
                "Below are research sources, each starting with its number like [12] and "
                "the number of sentences its summary may have.\n"
                "For EACH source write ONE line: its number, then its concrete findings IN "
                "ENGLISH (translate non-English sources): one sentence per distinct finding, "
                "with its numbers, who found or says it, and the sample or scope. Use up to "
                "the number of sentences its label allows; a source with fewer distinct "
                "findings gets fewer sentences. Example:\n"
                "[12] A 2024 survey of 1,200 nurses in Spain by the national nursing council "
                "found 40% work unpaid overtime every week. Overtime was highest in public "
                "hospitals. The council links it to unfilled posts but gives no figure.\n"
                f"If a source has nothing usable (navigation, ads, error page), "
                f"write: [12] {NO_EVIDENCE}\n"
                "One line per source, nothing else.\n\n" + batch),
            api_key, model,
        )
        wanted = {s["idx"]: s for s in todo}
        for line in "\n".join(answers).splitlines():
            m = re.match(r"^\s*\**\[(\d+)\]\**\s*[:.\-]?\s*(.+)$", line)
            if m and int(m.group(1)) in wanted:
                s = wanted[int(m.group(1))]
                cache[str(s["idx"])] = {"key": key(s), "card": m.group(2).strip()}
        todo = [s for s in todo if cache.get(str(s["idx"]), {}).get("key") != key(s)]

    if todo:
        print(f"   {len(todo)} sources still have no card (AI calls failed or skipped them)")

    # Cards the AI left in the source's language (mostly Japanese, Chinese,
    # Korean or Russian script) are translated; a translation must keep the numbers
    def foreign(card):
        letters = re.findall(r"[^\W\d_]", card) or ["x"]
        return len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3\u0400-\u04ff]", card)) / len(letters) > 0.3
    redo = [k for k, v in cache.items() if foreign(v["card"])]
    if redo:
        print(f"   Translating {len(redo)} cards written in their source's language...")
    for start in range(0, len(redo), 40):
        batch = redo[start:start + 40]
        reply = call_ai("Translate each line into English. Keep every number and name. One line "
                        "each, starting with its label:\nT1: <translation>\n\n"
                        + "\n".join(f"T{i}: {cache[k]['card']}" for i, k in enumerate(batch, 1)),
                        api_key, model, temp=0) or ""
        for m in re.finditer(r"^\W*T(\d+)\W*:\s*(.+)$", reply, re.M):
            i = int(m.group(1)) - 1
            if 0 <= i < len(batch):
                old_card, new_card = cache[batch[i]]["card"], m.group(2).strip()
                if set(re.findall(r"\d+", old_card)) <= set(re.findall(r"\d+", new_card)):
                    cache[batch[i]]["card"] = new_card

    Path(corpus_dir).mkdir(exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    return {s["idx"]: cache[str(s["idx"])]["card"]
            for s in corpus if str(s["idx"]) in cache}


# The source types, as the AI is asked to choose them (labels as classify_source writes them)
SOURCE_TYPES = {
    "vendor/primary": "a company writing about AI or software tools it makes or sells",
    "consulting": "a consultancy or market-research firm",
    "journalism": "news media or a magazine",
    "academic": "a university, research paper or journal",
    "government": "a government body",
    "intl institution": "an international organisation (OECD, UN, EU...)",
    "experiential/forum": "a forum, Q&A site or community discussion",
    "experiential/blog": "a personal or developer blog, or an individual's tutorial",
    "SEO risk": "a content farm, listicle, affiliate or review site made to rank in search",
}
TYPES_FILE = "source_types.json"


def type_sources(corpus, cards, api_key, model, corpus_dir=SCRATCHPAD):
    """
    Give a type to each source the domain rules left "unclassified": the AI chooses
    from SOURCE_TYPES by the site, title and evidence card. Cached in
    corpus_dir/source_types.json; sets s["quality"] and returns how many changed.
    """
    path = os.path.join(corpus_dir, TYPES_FILE)
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = [s for s in corpus if s.get("quality") == "unclassified" and s.get("url") not in cache]
    for start in range(0, len(todo), 60):
        batch = todo[start:start + 60]
        reply = call_ai(
            "For each source below, choose who published it, from these types:\n"
            + "\n".join(f"{t}: {d}" for t, d in SOURCE_TYPES.items())
            + "\nunclassified: none fits, or you cannot tell\n"
            "Answer one line per source: S1: <type>\n\n"
            + "\n".join(f"S{i}: {s.get('domain', '')} | {s.get('title', '')[:100]} | "
                         f"{cards.get(s['idx'], '')[:200]}" for i, s in enumerate(batch, 1)),
            api_key, model, temp=0) or ""
        for m in re.finditer(r"^\W*S(\d+)\W*:\s*(.+?)\s*$", reply, re.M):
            i, label = int(m.group(1)) - 1, m.group(2).strip().strip("*").lower()
            match = next((t for t in SOURCE_TYPES if t.lower() == label), None)
            if 0 <= i < len(batch) and match:
                cache[batch[i]["url"]] = match
    if todo:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, ensure_ascii=False)
    changed = 0
    for s in corpus:
        if s.get("quality") == "unclassified" and cache.get(s.get("url")):
            s["quality"] = cache[s["url"]]
            changed += 1
    return changed


def load_evidence_cards(corpus_dir=SCRATCHPAD):
    """Return the cached cards as {idx: card text} ({} if none were built)."""
    cache_path = os.path.join(corpus_dir, CARDS_FILE)
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, encoding="utf-8") as f:
        return {int(k): v["card"] for k, v in json.load(f).items()}
