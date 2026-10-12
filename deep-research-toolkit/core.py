"""
core.py: the shared functions that every mode uses.
====================================================
This file contains the reusable functions:
  - Searching the web (Google/DuckDuckGo/Brave/SearXNG)
  - Reading web pages (extracting article text)
  - Talking to AI models (via OpenRouter API)
  - Saving/loading work (the scratchpad system)
  - Classifying sources (academic? vendor? forum?)
  - Detecting what language a text is in
  - Configuration (which languages to search, which phrases to ban)

Every mode imports from this file, so no mode runs without it.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS: built-in Python tools (no install needed)
# ════════════════════════════════════════════════════════════════════════

import json       # Read and write JSON files. JSON is a text format for structured data,
                  # for example {"name": "Alice", "age": 30}

import re         # "Regular expressions": patterns for searching text.
                  # For example, re.sub(r"\s+", " ", text) replaces multiple spaces with one.

import sys        # Stop with a clear message when a required package is missing
import os         # Work with files and folders: check if a file exists, create folders,
                  # read environment variables (system-wide settings like your API key)

import importlib.util  # Check whether optional packages are installed without
                       # importing them (used for the citationlib check below)

import requests   # The most widely used library for web requests. When you visit a URL,
                  # your browser sends an "HTTP request." This library does the same
                  # from Python code: requests.get("https://example.com")

from urllib.parse import urlparse  # Break a URL into parts:
                                   # urlparse("https://www.bbc.com/news/tech")
                                   #   → scheme="https", netloc="www.bbc.com", path="/news/tech"
                                   # We use this to extract the domain name from URLs.

from pathlib import Path  # Handles file paths as objects instead of plain strings.
                          # Path("folder").mkdir(exist_ok=True) creates a folder if it
                          # doesn't exist, and does nothing if it already does.


# ════════════════════════════════════════════════════════════════════════
# LIBRARY LOADING: try to import each optional library
# ════════════════════════════════════════════════════════════════════════
#
# install.py installs each library below, but some installs can fail (no
# internet connection, unsupported OS, etc.). So each import uses try/except:
#   try:    → try to load the library
#   except: → if it fails, record that it's missing and move on
#
# This way the toolkit still runs when some libraries are missing, and it
# skips the features that need them.
#
# We use "flag variables" (like _TRAF = True/False) so other parts of
# the code can check "is trafilatura available?" before trying to use it.

# --- Web search (ddgs multi-engine primary, Google fallback) ---
# ddgs is the primary search engine: free, no API key. Despite the name it
# queries several engines (Bing, Brave, DuckDuckGo, Google, Yandex, ...) and
# merges the results. It replaces the old "duckduckgo-search" package, which
# was renamed to ddgs and now returns zero results (DuckDuckGo blocks it)
# without raising any error. We still accept the old package as a fallback.
try:
    from ddgs import DDGS  # DDGS = Dux Distributed Global Search
    _SEARCH_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # Old name; likely returns nothing
        _SEARCH_DDG = True
    except ImportError:
        _SEARCH_DDG = False

# multi_search_engine is the fallback: if DDG returns nothing (which happens
# with some non-English queries), we try Google via this package.
# The package installs as multi_search_engine but is imported as
# multi_search (the two names differ).
# Google's index holds many more Japanese, Chinese and Korean pages, so this
# fallback matters for the toolkit's searches in 9 languages.
try:
    from multi_search import search as _google_search
    _SEARCH_GOOGLE = True
except ImportError:
    _SEARCH_GOOGLE = False

# Set overall search availability flag
if _SEARCH_DDG or _SEARCH_GOOGLE:
    _SEARCH = "available"
else:
    _SEARCH = "none"  # No search library. Install one with: pip install ddgs

# --- Trafilatura (article text extraction from web pages) ---
try:
    import trafilatura  # Extracts the article text from a web page,
    from trafilatura.settings import use_config as _traf_config
                        # ignoring navigation, ads, footers, sidebars.
                        # It scored best at this task in benchmarks.
    _TRAF = True
except ImportError:
    _TRAF = False

# Give up on a page after 15 seconds instead of trafilatura's default of 30,
# so one slow site doesn't hold up a batch of downloads.
if _TRAF:
    _TRAF_CONFIG = _traf_config()
    _TRAF_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "15")
    # Some pages always fail to download (the toolkit counts them). Without
    # these lines, trafilatura logs several lines per failed page and the
    # progress messages are hard to find among them.
    import logging
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

# --- Camoufox (stealth browser for bot-protected sites) ---
try:
    from camoufox.sync_api import Camoufox  # A modified Firefox browser that
                                             # websites take for a person browsing.
                                             # It gets past anti-scraping blocks.
    from camoufox.pkgman import installed_verstr
    installed_verstr()   # Raises if the browser itself (~300MB) isn't downloaded
    _CAMO = True
except Exception:        # Not installed, browser missing, or a broken install
    _CAMO = False

# --- Language detection ---
try:
    from langdetect import detect as _detect_lang  # Takes text and returns its
                                                    # language: "en", "fr", "ja"
    _LANG = True
except ImportError:
    _LANG = False
    # Create a dummy function that always returns "unknown"
    def _detect_lang(t):
        return "unknown"

# --- Semantic Scholar (academic paper search, 200M+ papers) ---
try:
    from semanticscholar import SemanticScholar  # Search academic papers by keyword,
                                                  # get citation counts, abstracts, etc.
    _S2 = True
except ImportError:
    _S2 = False

# --- arXiv (preprint search: the site where scientists post new research) ---
try:
    import arxiv as _arxiv  # "as _arxiv" renames it to avoid conflicts.
                            # arXiv hosts millions of research papers, mostly
                            # in physics, math, and computer science.
    _ARXIV = True
except ImportError:
    _ARXIV = False

# --- Hacker News (developer discussion forum, searched via the Algolia API) ---
try:
    from hn import search_by_date as _hn_search  # Search HN stories and comments
                                                   # by keyword, date, and points.
    _HN = True
except ImportError:
    _HN = False

# --- Citation formatting (generates APA references from URLs) ---
# Check whether citationlib is installed without importing it.
# importlib.util.find_spec returns a spec object if the package exists,
# or None if it doesn't. The package is not imported, so none of its code runs.
_CITE = importlib.util.find_spec("citationlib") is not None


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION: settings used across all modes
# ════════════════════════════════════════════════════════════════════════

# Where all intermediate work is saved. Every mode reads from and writes
# to this folder, so they can share data without passing it directly.
SCRATCHPAD = "scratchpad"

# The 9 languages the toolkit searches in. For each language the AI writes
# its own queries, phrased the way a native speaker would ask, instead of
# translating the English queries.
#
# "why" says what perspective this language adds that English sources miss.
# explore puts it in the prompt that asks for queries, because developer
# culture differs from one country to the next.
#
# "region" tells the search engines which country's results to favor.
# Without it every search runs as US-English, and e.g. Chinese queries
# came back as mostly English pages (github.com, lexology.com) instead of
# Zhihu, CSDN or Toutiao.
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

# Phrases the voice cleaner will strip from AI-generated text.
# These are "LLM-speak": filler phrases that AI models overuse.
# They make text sound generic and untrustworthy in academic writing.
# The voice cleaner in synthesize.py searches for these and removes them.
BANNED_PHRASES = [
    # Hedging filler (says nothing, wastes the reader's time)
    "it's important to note", "it is important to note",
    "it's worth noting", "it is worth noting",
    "in today's rapidly evolving", "in an era of",
    "in the rapidly evolving landscape",
    # Buzzwords (sound impressive, mean nothing specific)
    "paradigm shift", "game-changer", "game changer",
    "revolutionize", "revolutionizing", "transformative potential",
    "unprecedented", "cutting-edge", "groundbreaking",
    "deep dive", "delve", "delving into",
    "holistic approach", "robust framework",
    "navigate the complexities", "double-edged sword",
    "the landscape of", "at the forefront",
    # Therapy-speak (inappropriate for academic writing)
    "and that's okay", "embrace the change", "lean into",
    # Hollow transitions (don't add meaning between paragraphs)
    "furthermore", "moreover", "additionally",
    "in conclusion", "to summarize",
    "as we have seen", "as discussed above",
    # Metaphors and idioms (unclear to non-native English speakers)
    "tip of the iceberg", "opens the door",
    "paves the way", "silver bullet",
    "low-hanging fruit", "move the needle",
    "reshaping the landscape", "spearheading",
    "synergy", "leverage the power",

    # ── Non-English LLM-speak ───────────────────────────────────────
    # AI models produce filler in other languages too.
    # These are the French, German, Japanese, Spanish, and Portuguese
    # equivalents of the English patterns above.
    #
    # French (common in Claude/GPT French output)
    "il est important de noter", "il convient de souligner",
    "dans un monde en constante évolution", "force est de constater",
    "en effet", "par ailleurs", "en conclusion",
    "un véritable changement de paradigme",
    # German
    "es ist wichtig zu beachten", "hervorzuheben ist",
    "in der heutigen schnelllebigen", "zusammenfassend lässt sich sagen",
    "darüber hinaus", "des Weiteren",
    # Japanese (LLM-speak patterns in Japanese output)
    "重要なのは", "注目すべきは",  # "what's important is" / "what's noteworthy is"
    "言うまでもなく",              # "needless to say"
    "まとめると",                  # "to summarize" (hollow transition)
    # Spanish
    "es importante destacar", "cabe señalar",
    "en un mundo en constante evolución", "en conclusión",
    # Portuguese
    "é importante notar", "vale ressaltar",
    "em um mundo em constante evolução", "em conclusão",
]


# ════════════════════════════════════════════════════════════════════════
# SEARCH: functions that find things on the internet
# ════════════════════════════════════════════════════════════════════════

def init_search(serper_key=None, brave_key=None):
    """
    Set up the search engine. Called once at the start of explore/deepen.
    Currently uses ddgs (free, no key needed).
    The serper_key/brave_key parameters are kept for future expansion
    but are not used with the current DDG-only setup.
    """
    pass  # ddgs needs no setup


def search_web(query, n=5, region="us-en"):
    """
    Search the web for a query and return up to n results.
    region (e.g. "jp-jp", see LANGUAGES) favors results from that country.

    Each result is a dictionary with keys:
      - "title":   the page title
      - "url":     the full URL
      - "snippet": a preview of the page content
      - "domain":  the domain name alone (e.g., "bbc.com")

    Two-tier search:
      1. ddgs (primary: free, no API key, queries several engines)
      2. Google via multi_search_engine (fallback, used when DDG returns
         nothing, which happens with some non-English queries. Google's index
         has more Japanese, Chinese and Korean pages.)

    Returns an empty list if both engines fail.
    """
    results = []  # Will hold our results

    # ── Tier 1: ddgs multi-engine search (primary) ──────────────────
    # Two attempts: ddgs picks engines at random, so a query that hit only
    # slow/blocked engines (it raises a timeout) often works on a retry.
    for attempt in range(2):
        if not _SEARCH_DDG or results:
            break
        try:
            # .text() searches for text results
            for r in DDGS().text(query, region=region, max_results=n):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),  # DDG uses "href"
                    "snippet": r.get("body", ""),              # DDG uses "body"
                })
        except Exception as e:
            # Print the error (after the second failure) instead of hiding it.
            if attempt == 1:
                print(f"      Web search error: {type(e).__name__}: {e}")

    # ── Tier 2: Google fallback (when DDG returns empty) ────────────
    # DDG sometimes returns nothing for non-English queries or narrow
    # searches. Google's index has many more pages, especially in
    # Japanese, Chinese, Korean and Russian.
    # Google is tried only when DDG returned zero results. Running it on
    # every search as well would add duplicate results.
    if not results and _SEARCH_GOOGLE:
        try:
            # _google_search is the search() function from multi_search.
            # search_engine="google" uses Google without an API key (scrapes HTML).
            # The return format is a list, but the structure of each item
            # varies, so the code handles both dict-style and tuple-style results.
            raw = _google_search(query, num_results=n, search_engine="google")

            for r in (raw or [])[:n]:
                if isinstance(r, dict):
                    # Dict-style result: {"title": ..., "link": ..., "snippet": ...}
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", r.get("url", r.get("href", ""))),
                        "snippet": r.get("snippet", r.get("description", "")),
                    })
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    # Tuple-style result: (title, url) or (title, url, snippet)
                    results.append({
                        "title": str(r[0]),
                        "url": str(r[1]),
                        "snippet": str(r[2]) if len(r) > 2 else "",
                    })

            if results:
                print(f"      DDG empty → used Google fallback ({len(results)} results)")

        except Exception as e:
            print(f"      Google fallback error: {type(e).__name__}: {e}")

    # ── Add domain name to each result ──────────────────────────────
    # urlparse breaks a URL into parts. .netloc is the domain.
    # "https://www.bbc.com/news/tech" → netloc = "www.bbc.com"
    for r in results:
        r["domain"] = urlparse(r.get("url", "")).netloc

    # Drop results without a usable web address (some engines sometimes
    # return an empty or relative link, which can't be fetched)
    return [r for r in results if r["domain"] and r["url"].startswith("http")]


def search_academic(query, n=5):
    """
    Search for academic papers using Semantic Scholar and arXiv.

    Returns a list of papers, each with:
      - "title":    paper title
      - "url":      link to the paper
      - "snippet":  the abstract (summary of the paper)
      - "domain":   which database it came from
      - "year":     publication year
      - "citations": how many times other papers cited it (S2 only)
    """
    papers = []

    # ── Semantic Scholar (200M+ papers, strong in CS/AI) ────────────
    if _S2:
        try:
            # Create a Semantic Scholar client and search
            # retry=False: without an API key Semantic Scholar often answers
            # "too many requests", and its built-in retries back off for
            # minutes per query. It fails at once instead, and arXiv still runs.
            for p in SemanticScholar(retry=False, timeout=20).search_paper(query, limit=n):
                papers.append({
                    "title": p.title or "",
                    "domain": "semanticscholar.org",
                    # Build a URL to the paper's page
                    "url": p.url or f"https://www.semanticscholar.org/paper/{p.paperId}",
                    # The whole abstract, used as the paper's content
                    "snippet": p.abstract or "",
                    # getattr(p, "year", None) means: get p.year if it exists,
                    # otherwise return None (nothing). Safer than p.year which
                    # would crash if the paper doesn't have a year field.
                    "year": getattr(p, "year", None),
                    "citations": getattr(p, "citationCount", None),
                })
        except Exception as e:
            print(f"      Semantic Scholar error: {type(e).__name__}: {e}")

    # ── arXiv (preprints: recent, not yet peer-reviewed) ────────────
    if _ARXIV:
        try:
            client = _arxiv.Client()   # Create an arXiv API client
            # Build a search query and get results
            search = _arxiv.Search(query=query, max_results=n,
                                   sort_by=_arxiv.SortCriterion.Relevance)
            for p in client.results(search):
                papers.append({
                    "title": p.title,
                    "domain": "arxiv.org",
                    "url": p.entry_id,           # arXiv's unique URL for this paper
                    "snippet": p.summary or "",   # The whole abstract
                    # p.published is a datetime object; .year extracts only the year
                    "year": p.published.year if p.published else None,
                })
        except Exception as e:
            print(f"      arXiv error: {type(e).__name__}: {e}")

    return papers


def search_hn(query, n=10):
    """
    Search Hacker News for developer discussions about a topic.

    Hacker News is a forum where developers discuss tech. It's run by
    Y Combinator (a well-known startup incubator). The discussions are often
    high-quality and show developers' own opinions, which can differ from
    what the companies selling the tools say.

    Returns a list of stories with title, URL, snippet, and upvote count.
    """
    results = []

    if _HN:
        try:
            # search_by_date searches HN via the Algolia API
            # stories=True means only return stories (not comments)
            # created_at__gt="2024-01-01" means only posts after Jan 1, 2024
            for item in _hn_search(query, stories=True, created_at__gt="2024-01-01"):
                if len(results) >= n:
                    break  # Stop once we have enough results

                results.append({
                    "title": item.get("title") or "",
                    # Build the HN URL from the item's ID
                    "url": f"https://news.ycombinator.com/item?id={item.get('objectID', '')}",
                    # story_text is null (not missing) for link posts, so
                    # .get("story_text", "") would return None here
                    "snippet": item.get("story_text") or "",
                    "domain": "news.ycombinator.com",
                    "points": item.get("points", 0),  # Upvotes (popularity)
                })
        except Exception as e:
            print(f"      HN search error: {type(e).__name__}: {e}")
    else:
        # If python-hn isn't installed, fall back to web-searching HN
        # "site:news.ycombinator.com" tells the search engine to only
        # return results from that specific website
        results = search_web(f"site:news.ycombinator.com {query}", n=n)

    return results


# ════════════════════════════════════════════════════════════════════════
# CONTENT EXTRACTION: reading the text of web pages
# ════════════════════════════════════════════════════════════════════════

# How much text to keep per source. Measured on 40 real corpus pages: median
# 11,300 characters, 90th percentile 28,400. The old 3,000-character limit
# threw away 79% of the text (often everything after the introduction);
# 8,000 still lost 51%. 30,000 keeps 93% and only cuts the longest pages.
# Everything downstream (cards, analyze, the brief) reads all of what's kept.
CONTENT_CHARS = 30000


def fetch_page(url, max_chars=CONTENT_CHARS, use_browser=True):
    """
    Visit a web page and extract its main article text.
    use_browser=False skips the (slow, one-at-a-time) camoufox tier.

    This is a three-tier system:
      Tier 1: trafilatura (fast, no browser, works ~80% of the time)
      Tier 2: camoufox (stealth browser, gets past bot detection, slower)
      Tier 3: give up and return None

    Returns the extracted text (up to max_chars), or None if it fails.

    Reason for two tiers: most websites serve their content as plain HTML,
    which trafilatura can read directly. Some sites (Reddit, LinkedIn,
    news paywalls) detect automated access and block it. Camoufox opens
    a real browser that these sites take for a person, so they let it through.
    """

    # ── Tier 1: trafilatura (fast, no browser needed) ───────────────
    if _TRAF:
        try:
            # Step 1: Download the raw HTML of the page
            html = trafilatura.fetch_url(url, config=_TRAF_CONFIG)

            if html:
                # Step 2: Extract the article text from the HTML
                # include_tables=True: keep data tables (useful for research)
                # favor_recall=True: keep more text, even if some non-article text comes with it
                text = trafilatura.extract(html, include_tables=True, favor_recall=True)

                # Return it only if it has more than 100 characters.
                # A result of 10 characters probably means extraction failed.
                if text and len(text.strip()) > 100:
                    return text[:max_chars]  # Truncate to max length
        except Exception:
            pass  # trafilatura failed; go on to Tier 2

    # ── Tier 2: camoufox (stealth browser, for protected sites) ─────
    if _CAMO and use_browser:
        try:
            # "with Camoufox(...) as browser" opens a stealth Firefox browser.
            # headless=True means invisible (no window pops up on your screen).
            # When the "with" block ends, the browser closes automatically.
            with Camoufox(headless=True) as browser:
                return _read_with_browser(browser, url, max_chars)
        except Exception as e:
            print(f"      camoufox extraction error: {type(e).__name__}")

    # ── Tier 3: give up ─────────────────────────────────────────────
    return None  # Couldn't read the page with any method


# JavaScript run inside the page to get its main text: remove navigation,
# headers, footers and sidebars, then take the main content area.
_EXTRACT_JS = """() => {
    for (const s of ['nav','header','footer','aside','.sidebar'])
        document.querySelectorAll(s).forEach(e => e.remove());
    const m = document.querySelector('main,article,[role="main"]');
    return (m || document.body).innerText;
}"""


def _read_with_browser(browser, url, max_chars):
    """Open url in an already-running camoufox browser; return its text, or None."""
    page = browser.new_page()                  # Open a new tab
    try:
        page.goto(url, timeout=20000)          # 20 second timeout
        page.wait_for_load_state("domcontentloaded")
        text = page.evaluate(_EXTRACT_JS)
        if text and len(text.strip()) > 100:
            # Collapse runs of whitespace into single spaces
            return re.sub(r"\s+", " ", text).strip()[:max_chars]
        return None
    finally:
        page.close()


def fetch_pages(urls, max_chars=CONTENT_CHARS, workers=8):
    """
    Fetch many pages at once. Returns {url: text} for the pages that worked.

    Tier 1 (trafilatura) runs `workers` downloads in parallel. Most of the
    time goes to waiting on websites, so this is ~6x faster than one by
    one. Pages that fail are then retried one at a time with the camoufox
    browser (tier 2), which can't safely run in parallel.
    """
    # Without a page reader every page fails and only search snippets are
    # kept. Stopping here avoids a half-hour run that saves nothing.
    if urls and not (_TRAF or _CAMO):
        sys.exit("\n   No page reader is installed (trafilatura). "
                 "Run install.py, or: pip install trafilatura")

    import time
    started = time.time()
    urls = list(dict.fromkeys(urls))  # Drop duplicates, keep order
    print(f"      reading {len(urls)} pages ({workers} at a time)...")
    texts = _read_pages_isolated(urls, max_chars, workers)
    pages = {u: texts[u] for u in urls if texts.get(u)}

    # Retry the failures in a single browser (opening one per page is slow)
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
                        continue  # This page still won't load; skip it
        except Exception as e:
            print(f"      camoufox unavailable: {type(e).__name__}")

    print(f"      read {len(pages)} of {len(urls)} pages in {time.time() - started:.0f}s")
    return pages


def _read_pages_isolated(urls, max_chars, workers):
    """
    Read pages with trafilatura, `workers` at a time, in a helper process.
    Returns {url: text or None}.

    Reason for a helper process: a crash inside a C library (the HTML parser
    lxml/libxml2, with errors such as "double free or corruption" or
    "segmentation fault") kills the whole program it happens in. It once
    ended an unattended run two minutes into explore. Here it kills only the
    helper: the few pages it was reading at that moment are skipped, and a
    new helper continues with the rest.
    """
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
        reading = set()   # Pages the helper started but hasn't finished
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # A line some library printed on its own; skip it
            if "start" in msg:
                reading.add(msg["start"])
            elif "done" in msg:
                reading.discard(msg["done"])
                results[msg["done"]] = msg["text"]
        return reading, proc.wait()

    results = {}
    jobs = [(list(urls), workers)]   # (pages to read, how many at a time)
    crashes = 0
    while jobs and crashes < 10:
        todo, n_workers = jobs.pop(0)
        todo = [u for u in todo if u not in results]
        if not todo:
            continue
        reading, exit_code = run_helper(todo, n_workers)
        if exit_code == 0:
            continue
        crashes += 1
        rest = [u for u in todo if u not in results and u not in reading]
        if n_workers == 1 and reading:
            # The helper read one page at a time, so this page caused the crash. Skip it.
            results.update(dict.fromkeys(reading))
            print(f"      skipped a page that crashes the page reader: {next(iter(reading))[:80]}",
                  flush=True)
        elif reading:
            # One of these crashed it: read them again one at a time to find which
            print(f"      the page reader crashed (exit code {exit_code}) -- re-reading the "
                  f"{len(reading)} pages it was on one at a time, then the other {len(rest)}",
                  flush=True)
            jobs.insert(0, (sorted(reading), 1))
        else:
            # It crashed before reading any page (possibly a broken install). Give up.
            print(f"      the page reader won't start (exit code {exit_code})", flush=True)
            break
        if rest:
            jobs.insert(1 if reading and n_workers > 1 else 0, (rest, n_workers))
    return results


def _read_pages_worker():
    """
    The helper process of _read_pages_isolated: reads the job (JSON) from
    stdin and prints one JSON line per page: "start" before reading it and
    "done" with its text afterwards. If this process crashes, the main
    program then knows which pages were being read at that moment.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    job = json.loads(sys.stdin.read())
    lock = threading.Lock()

    def emit(msg):
        with lock:   # One whole line at a time, even from 8 threads
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    def read_one(url):
        emit({"start": url})
        emit({"done": url, "text": fetch_page(url, job["max_chars"], use_browser=False)})

    with ThreadPoolExecutor(max_workers=job["workers"]) as pool:
        list(pool.map(read_one, job["urls"]))


def region_for(query):
    """
    Search region for a query, from its script: a Korean query searches
    Korean pages, a Japanese one Japanese pages, etc. Latin-script queries
    (English, French, German...) can't be told apart reliably this way, so
    they use the US-English region.
    """
    for lang, script in [("ko", r"[\uac00-\ud7a3]"),     # Hangul
                         ("ja", r"[\u3040-\u30ff]"),     # Hiragana/Katakana (before zh:
                         ("zh", r"[\u4e00-\u9fff]"),     #   Japanese also uses Kanji)
                         ("ru", r"[\u0400-\u04ff]")]:    # Cyrillic
        if re.search(script, query):
            return LANGUAGES[lang]["region"]
    return "us-en"


def search_many(jobs, workers=3):
    """
    Run several web searches at once. jobs is a list of (query, n, region);
    returns a list of result lists in the same order. Kept to a few at a
    time so the search engines don't block us.
    """
    from concurrent.futures import ThreadPoolExecutor  # Run searches in parallel

    # Without a search library every search finds nothing. Stop and print
    # the reason instead of reporting 0 results and continuing.
    if jobs and _SEARCH == "none":
        sys.exit("\n   No web search library is installed (ddgs). "
                 "Run install.py, or: pip install ddgs")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: search_web(job[0], n=job[1], region=job[2]), jobs))


# ════════════════════════════════════════════════════════════════════════
# AI: sending prompts to AI models through the OpenRouter API
# ════════════════════════════════════════════════════════════════════════

def call_ai(prompt, api_key, model, temp=0.6, system="", max_tokens=16000):
    """
    Send a prompt to an AI model and get back its response.

    Every AI step in the toolkit goes through this function.
    It sends a message to a model (Claude, GPT, Llama, Mistral, etc.)
    through the OpenRouter API and returns the model's reply.

    Parameters:
        prompt:     what you want the AI to do (the question/instruction)
        api_key:    your OpenRouter API key (it works as a password)
        model:      which AI model to use (e.g., "google/gemini-2.5-flash-lite")
        temp:       "temperature", which controls randomness. 0 = most
                    predictable, 1 = most varied. 0.6 is a middle setting
                    that suits research.
        system:     optional "system prompt" that tells the AI how to behave
                    (e.g., "you are an academic researcher, be precise")
        max_tokens: cap on the length of the reply. The default (16,000
                    tokens, ~12,000 words) is far above any normal answer,
                    and it stops a model stuck in a loop: one once wrote
                    270 KB of made-up search queries into the bias audit.

    Returns the AI's response as a text string, or "" if something went wrong.

    How an API call works:
    Your code sends a message over the internet to OpenRouter's server.
    OpenRouter forwards it to the AI model you chose.
    The model generates a response.
    OpenRouter sends that response back to your code.
    The whole trip takes 2-30 seconds, depending on how much you asked for.
    """

    # Build the messages list. AI APIs use a "conversation" format:
    # each message has a "role" (who's talking) and "content" (what they said).
    msgs = []

    # The "system" message sets the AI's behavior (optional)
    if system:
        msgs.append({"role": "system", "content": system})

    # The "user" message is your question/instruction
    msgs.append({"role": "user", "content": prompt})

    import time

    # Temporary failures ("too many requests" (429), a busy or timed-out
    # server (5xx), a dropped connection) are retried after waits of 5, 20
    # and 60 seconds, so an unattended run doesn't lose a step to a
    # 20-second outage.
    # Permanent ones (bad key, no credit left, bad request) fail at once.
    waits = [5, 20, 60]
    for attempt in range(len(waits) + 1):
        try:
            # requests.post sends data to a web server: here, OpenRouter's
            # chat endpoint, with the API key and the conversation as JSON
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",   # Standard key format
                    "Content-Type": "application/json",     # We're sending JSON
                },
                json={
                    "model": model,           # Which AI model to use
                    "messages": msgs,         # The conversation so far
                    "temperature": temp,      # How random/creative to be
                    **({"max_tokens": max_tokens} if max_tokens else {}),
                },
                timeout=180,  # Wait up to 3 minutes for a response (AI can be slow)
            )

            temporary = r.status_code == 429 or r.status_code >= 500
            if temporary and attempt < len(waits):
                wait = waits[attempt]
                if r.headers.get("Retry-After", "").isdigit():   # The server says how long
                    wait = min(int(r.headers["Retry-After"]), 120)
                print(f"   AI busy ({r.status_code}) -- retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue

            # An error reply (400, 401, 402...) raises here and goes to "except"
            r.raise_for_status()

            # The AI's text is nested inside: choices -> [first] -> message -> content
            # (a model can return an empty reply as null, which becomes "")
            return r.json()["choices"][0]["message"]["content"] or ""

        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < len(waits):
                print(f"   AI connection problem ({type(e).__name__}) -- retrying in "
                      f"{waits[attempt]}s", flush=True)
                time.sleep(waits[attempt])
                continue
            print(f"   AI error: {e}")
            return ""
        except Exception as e:
            # Something went wrong (bad API key, no credit, bad request, etc.)
            print(f"   AI error: {e}")
            return ""  # Return empty string so the calling code doesn't crash
    return ""


# ════════════════════════════════════════════════════════════════════════
# SCRATCHPAD: saving and loading intermediate work
# ════════════════════════════════════════════════════════════════════════
#
# The scratchpad is a folder where every mode saves its output.
# explore writes its findings there, analyze reads them and adds NLP
# results, stats reads those and adds statistical tests, and synthesize
# reads everything and writes the thesis. Each mode reads the previous
# mode's output.

def pad(name, content):
    """
    Save a piece of work to the scratchpad folder as a .md file.

    Parameters:
        name:    filename (without .md extension)
        content: the text to save

    Example: pad("01_definitions", "# Definitions\n\n...")
    Creates: scratchpad/01_definitions.md
    """
    # Create the scratchpad folder if it doesn't exist yet.
    # exist_ok=True means "don't crash if it already exists."
    Path(SCRATCHPAD).mkdir(exist_ok=True)

    # Build the full file path: scratchpad/01_definitions.md
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    # Open the file for writing ("w") with UTF-8 encoding
    # (UTF-8 handles all languages: Japanese, Chinese, Arabic, etc.)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)  # Write the content to the file

    print(f"   {p}")  # Tell the user where we saved it


def load_pad(name):
    """
    Load a previously saved scratchpad file.

    Returns the file contents as a string, or "" if the file doesn't exist.
    """
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    if os.path.exists(p):                           # Does the file exist?
        with open(p, encoding="utf-8") as f:        # Open for reading
            return f.read()                         # Return its entire contents
    return ""  # File doesn't exist → return empty string


def load_sources():
    """
    Load the source inventory from a previously saved metadata JSON file.

    The explore mode saves all source metadata to a JSON file.
    This function finds and loads the most recent one.
    """
    # Path(".").glob("research_*_meta.json") finds all files in the current
    # folder that match the pattern "research_*_meta.json" (the * matches anything).
    # sorted(..., reverse=True) puts the newest first (alphabetically, which
    # works because the filenames contain timestamps).
    for fn in sorted(Path(".").glob("research_*_meta.json"), reverse=True):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)  # json.load reads a JSON file into a Python dictionary

    return None  # No metadata file found


def load_corpus_texts():
    """
    Load all scratchpad .md files as a list of dictionaries.

    Each dictionary has:
      - "file": the filename
      - "text": the file's contents

    Used by ask_agent to build context for the AI.
    """
    texts = []
    for fn in sorted(Path(SCRATCHPAD).glob("*.md")):  # Find all .md files
        with open(fn, encoding="utf-8") as f:
            texts.append({"file": fn.name, "text": f.read()})
    return texts


# ════════════════════════════════════════════════════════════════════════
# SOURCE CLASSIFICATION: labelling each source by type
# ════════════════════════════════════════════════════════════════════════
#
# Not all sources are equally trustworthy. An academic paper from arxiv.org
# is more reliable than a blog post on medium.com. This function looks at
# the domain name and assigns a quality label.
#
# These labels are used in analysis (do vendor sources have different
# sentiment than academic ones?) and in the report (label vendor-funded
# studies as such).

# Plain names for the source types classify_source returns. Charts and the
# brief show these, so a reader doesn't have to decode "vendor/primary".
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
    """
    Given a domain name (like "arxiv.org" or "techcrunch.com"),
    return a quality label (like "academic" or "journalism").

    Uses pattern matching: if the domain contains "arxiv" anywhere,
    it's academic. If it contains "techcrunch", it's journalism.

    any(x in d for x in [...]) means:
    "is at least one of these strings found inside d?"
    It returns True if at least one match is found.
    """
    d = domain.lower()  # Lowercase so "ArXiv.org" and "arxiv.org" both match

    # Country-style endings (".edu", ".go.jp") are checked as suffixes (the
    # end of the name). Checking "edu" anywhere in the name would also match
    # e.g. "education-news.com".
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
        return "consulting"  # Consulting firms have their own biases

    # Companies that sell AI coding tools. Plain github.com is left out:
    # most of it is user repositories, which GitHub does not write itself.
    if any(x in d for x in ["github.blog", "docs.github.com", "microsoft.com", "openai.com",
                            "jetbrains", "stackoverflow.blog", "anthropic.com", "cursor.com",
                            "ibm.com", "tabnine", "codeium", "windsurf", "replit.com",
                            "sourcegraph", "kimi.ai", "aliyun.com"]):
        return "vendor/primary"  # These companies sell the tools being studied

    if any(x in d for x in ["news.ycombinator", "dev.to", "lobste.rs", "reddit.com", "qiita.com",
                            "zenn.dev", "note.com", "tistory.com", "brunch.co.kr", "velog.io",
                            "habr.com"]):
        return "experiential/forum"  # Discussions written by developers

    if any(x in d for x in ["reuters", "ft.com", "nytimes", "bloomberg", "techcrunch", "wired",
                            "lemonde", "lesechos", "heise.de", "golem.de", "zdnet", "theverge",
                            "arstechnica", "cio.com", "infoq", "itmedia", "nikkei", "sedaily",
                            "inews24", "vc.ru"]):
        return "journalism"

    if any(x in d for x in ["medium.com", "towardsdatascience", "analyticsvidhya"]):
        return "SEO risk"  # Often content farms / low-quality aggregators

    return "unclassified"  # Doesn't match any pattern


def detect_lang(text):
    """
    Detect what language a piece of text is written in.

    Returns a language code: "en" for English, "fr" for French,
    "ja" for Japanese, etc. Returns "unknown" if detection fails
    or the text is too short to detect reliably.

    It only tries when the text is longer than 50 characters. Shorter
    text doesn't give the detector enough to work with.
    """
    if _LANG and text and len(text) > 50:
        try:
            # Only analyze first 500 chars (faster). langdetect returns
            # regional codes like "zh-cn" / "zh-tw"; keep only the base
            # code so it matches the keys in LANGUAGES ("zh").
            return _detect_lang(text[:500]).split("-")[0]
        except Exception:
            pass  # Detection failed; return 'unknown'
    return "unknown"


# ════════════════════════════════════════════════════════════════════════
# CAPABILITIES: show what's installed and working
# ════════════════════════════════════════════════════════════════════════

def print_capabilities():
    """
    Print a checklist of which libraries are available.

    This runs at the start of explore/deepen so you can see at a glance
    what the toolkit can do. "OK" means the library is installed,
    "--" means it's missing (that feature will be skipped).
    """
    def ok(flag, missing="--"):
        return "OK" if flag else missing

    print(f"   Search:      DDG {ok(_SEARCH_DDG)} + Google fallback {ok(_SEARCH_GOOGLE)}")
    print(f"   Extract:     trafilatura {ok(_TRAF)} + camoufox {ok(_CAMO)}")
    print(f"   Academic:    S2 {ok(_S2)} + arXiv {ok(_ARXIV)}")
    print(f"   HN:          {ok(_HN, '(web fallback)')}")
    print(f"   Lang detect: {ok(_LANG)}")
    print(f"   APA:         {ok(_CITE, '(manual fallback)')}")


def extract_quoted_queries(text, min_len=10):
    """
    Pull search queries out of AI text where they appear in quotes.

    It matches every quoted string first (short ones included) and then
    filters by length. Matching only long strings (r'"([^"]{10,})"') pairs
    the wrong quote marks after a short quote: 'try "AI" or "junior dev
    adoption"' would yield ' or ' as a "query". Curly quotes are handled
    too, and a query never spans lines.
    """
    quoted = re.findall(r'["\u201c]([^"\u201c\u201d\n]*)["\u201d]', text or "")
    return [q.strip() for q in quoted if len(q.strip()) >= min_len]


# Ask the AI to mark its suggested searches this way (see QUERY_FORMAT use)
QUERY_FORMAT = ("Put each suggested search query on its own line, "
                "starting with 'QUERY: ' (no quotes around it).")


def extract_search_queries(text, min_len=10):
    """
    Pull the suggested search queries out of an AI critique / gap analysis.

    Prompts ask for lines like "QUERY: junior developer AI adoption 2025"
    (see QUERY_FORMAT). Grabbing every quoted string instead also grabbed
    quoted claims ("superficial", "cognitive load eroding,") and quoted
    source titles, which then got searched as if they were queries. Quoted
    text is only used as a fallback when the model ignored the format.
    Near-duplicates ("cognitive load eroding," vs "...eroding.") are dropped.
    """
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
# BATCHING: let the AI read more text than fits in one prompt
# ════════════════════════════════════════════════════════════════════════
#
# Used by ask_agent (answers from all the research), analyze (topic clustering)
# and synthesize (read every source before writing).

def make_batches(parts, batch_chars=12000):
    """
    Pack labelled text parts into batches of at most batch_chars characters.

    parts is a list of (label, text). Parts stay in order. A part longer than
    one batch is split into pieces, and each piece keeps its label
    ("── name (part 2) ──") so the AI still knows where it came from.
    """
    batches = []
    current = ""

    for label, text in parts:
        room = batch_chars - len(label) - 30  # Leave space for the header
        pieces = [text[i:i + room] for i in range(0, len(text), room)] or [""]

        for k, piece in enumerate(pieces, 1):
            name = label if len(pieces) == 1 else f"{label} (part {k})"
            block = f"── {name} ──\n{piece}\n"

            # Start a new batch if this block doesn't fit in the current one
            if current and len(current) + len(block) > batch_chars:
                batches.append(current)
                current = ""
            current += block

    if current:
        batches.append(current)
    return batches


def ai_map(batches, make_prompt, api_key, model, temp=0.2, workers=4):
    """
    Send one prompt per batch, several at a time, and return the answers
    in batch order. make_prompt(batch) builds each prompt. A failed call
    gives "" (call_ai already printed why). Lower workers if rate-limited.
    """
    from concurrent.futures import ThreadPoolExecutor  # Run calls in parallel

    def one(batch):
        return call_ai(make_prompt(batch), api_key, model, temp=temp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, batches))  # map keeps the original order


# ════════════════════════════════════════════════════════════════════════
# EVIDENCE CARDS: one short English summary per source
# ════════════════════════════════════════════════════════════════════════
#
# A card is 1-2 sentences of the concrete findings, numbers and claims in a
# source, written in English whatever the source's language, tagged with
# the source number: "[44] Korean survey: Copilot raised PR completions 26%..."
#
# Reason: the corpus is ~150 sources in 9 languages, ~400,000 characters.
# Cards make it ~10x smaller and still cover every source, so:
#   - synthesize can show the AI all sources instead of the first 10
#   - analyze can cluster sources by theme (clustering the raw multilingual
#     text gives topics labelled "die / der / de")
# Cards are cached in scratchpad/evidence_cards.json, so re-runs (e.g. after
# deepen adds sources) only pay for new or changed sources.

CARDS_FILE = "evidence_cards.json"
NO_EVIDENCE = "NO EVIDENCE"  # Card text for pages with nothing usable


def _content_key(item):
    """Fingerprint of a source's text, to notice when it changed."""
    import hashlib
    return hashlib.sha1(item.get("content", "").encode("utf-8")).hexdigest()[:12]


def build_evidence_cards(corpus, api_key, model, corpus_dir=SCRATCHPAD):
    """
    Return {idx: card text} for every source in corpus (a list of dicts with
    idx/title/domain/lang/content), reading sources ~8 at a time in parallel.
    Uses and updates the cache in corpus_dir/evidence_cards.json.
    """
    cache_path = os.path.join(corpus_dir, CARDS_FILE)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    # Sources with no card yet, or whose text changed since the card was made
    todo = [s for s in corpus
            if cache.get(str(s["idx"]), {}).get("key") != _content_key(s)]

    # Two passes: the second retries sources the AI skipped the first time
    for attempt in range(2):
        if not todo:
            break
        if attempt == 0:
            print(f"   Writing evidence cards for {len(todo)} sources...")
        parts = [(f"[{s['idx']}] ({s.get('lang', '?')}) {s.get('title', '')} -- {s.get('domain', '')}",
                  s.get("content", "")) for s in todo]   # the whole stored text
        answers = ai_map(
            # ~80,000 characters per call (~20k tokens): big enough that a
            # source is never split across calls (sources are <= CONTENT_CHARS)
            make_batches(parts, batch_chars=80000),
            lambda batch: (
                "Below are research sources about AI coding assistants, each "
                "starting with its number like [12].\n"
                "For EACH source write ONE line: its number, then 1-2 sentences "
                "IN ENGLISH with its concrete findings, numbers, claims and who "
                "says them (translate non-English sources). Example:\n"
                "[12] Stack Overflow survey of 65,000 developers: 76% use or plan "
                "to use AI tools, but only 43% trust their accuracy.\n"
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
                cache[str(s["idx"])] = {"key": _content_key(s), "card": m.group(2).strip()}
        todo = [s for s in todo if cache.get(str(s["idx"]), {}).get("key") != _content_key(s)]

    if todo:
        print(f"   {len(todo)} sources still have no card (AI calls failed or skipped them)")

    Path(corpus_dir).mkdir(exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    return {s["idx"]: cache[str(s["idx"])]["card"]
            for s in corpus if str(s["idx"]) in cache}


def load_evidence_cards(corpus_dir=SCRATCHPAD):
    """Return the cached cards as {idx: card text} ({} if none were built)."""
    cache_path = os.path.join(corpus_dir, CARDS_FILE)
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, encoding="utf-8") as f:
        return {int(k): v["card"] for k, v in json.load(f).items()}
