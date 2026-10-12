"""
core.py -- The shared toolbox that every mode uses.
====================================================
This file contains all the reusable functions:
  - Searching the web (Google/DuckDuckGo/Brave/SearXNG)
  - Reading web pages (extracting article text)
  - Talking to AI models (via OpenRouter API)
  - Saving/loading work (the scratchpad system)
  - Classifying sources (academic? vendor? forum?)
  - Detecting what language a text is in
  - Configuration (which languages to search, which phrases to ban)

No mode runs without this file. It's the foundation.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS -- built-in Python tools (no install needed)
# ════════════════════════════════════════════════════════════════════════

import json       # Read and write JSON files (a structured data format, like a spreadsheet
                  # but for computers -- {"name": "Alice", "age": 30})

import re         # "Regular expressions" -- a way to search for patterns in text.
                  # For example, re.sub(r"\s+", " ", text) replaces multiple spaces with one.

import os         # Work with files and folders: check if a file exists, create folders,
                  # read environment variables (system-wide settings like your API key)

import importlib.util  # Check whether optional packages are installed without
                       # actually importing them (used for citationlib check below)

import requests   # THE library for making web requests. When you visit a URL in your
                  # browser, your browser sends an "HTTP request." This library does the
                  # same thing from Python code: requests.get("https://example.com")

from urllib.parse import urlparse  # Break a URL into parts:
                                   # urlparse("https://www.bbc.com/news/tech")
                                   #   → scheme="https", netloc="www.bbc.com", path="/news/tech"
                                   # We use this to extract the domain name from URLs.

from pathlib import Path  # A nicer way to work with file paths than raw strings.
                          # Path("folder").mkdir(exist_ok=True) creates a folder if it
                          # doesn't exist, and does nothing if it already does.


# ════════════════════════════════════════════════════════════════════════
# LIBRARY LOADING -- try to import each optional library
# ════════════════════════════════════════════════════════════════════════
#
# Each library below was installed by install.py. But some might have failed
# to install (bad internet, wrong OS, etc.). So we use try/except:
#   try:    → try to load the library
#   except: → if it fails, note that it's missing and move on
#
# This way the toolkit still runs even if some libraries are missing --
# it just skips the features that need them.
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
        from duckduckgo_search import DDGS  # Old name -- likely returns nothing
        _SEARCH_DDG = True
    except ImportError:
        _SEARCH_DDG = False

# multi_search_engine is the fallback: if DDG returns nothing (which happens
# with some non-English queries), we try Google via this package.
# NOTE: the install name is multi_search_engine but the import name is
# multi_search (yes, they're different -- not ideal but it works).
# Google's index is significantly deeper for Japanese, Chinese, Korean content,
# so this fallback matters for the multilingual doctrine.
try:
    from multi_search import search as _google_search
    _SEARCH_GOOGLE = True
except ImportError:
    _SEARCH_GOOGLE = False

# Set overall search availability flag
if _SEARCH_DDG or _SEARCH_GOOGLE:
    _SEARCH = "available"
else:
    _SEARCH = "none"  # No search at all -- install: pip install ddgs

# --- Trafilatura (article text extraction from web pages) ---
try:
    import trafilatura  # Extracts the article text from a web page,
                        # ignoring navigation, ads, footers, sidebars.
                        # Best-in-class for this job (benchmarked).
    _TRAF = True
except ImportError:
    _TRAF = False

# --- Camoufox (stealth browser for bot-protected sites) ---
try:
    from camoufox.sync_api import Camoufox  # A modified Firefox browser that
                                             # looks like a real human browsing,
                                             # not a bot. Bypasses anti-scraping.
    _CAMO = True
except ImportError:
    _CAMO = False

# --- Language detection ---
try:
    from langdetect import detect as _detect_lang  # Feed it text, it tells you
                                                    # what language: "en", "fr", "ja"
    _LANG = True
except ImportError:
    _LANG = False
    # Create a dummy function that always returns "unknown"
    def _detect_lang(t):
        return "unknown"

# --- Semantic Scholar (academic paper search -- 200M+ papers) ---
try:
    from semanticscholar import SemanticScholar  # Search academic papers by keyword,
                                                  # get citation counts, abstracts, etc.
    _S2 = True
except ImportError:
    _S2 = False

# --- arXiv (preprint paper search -- where scientists post new research) ---
try:
    import arxiv as _arxiv  # "as _arxiv" renames it to avoid conflicts.
                            # arXiv hosts millions of research papers, mostly
                            # in physics, math, and computer science.
    _ARXIV = True
except ImportError:
    _ARXIV = False

# --- Hacker News (developer discussion forum -- via Algolia search API) ---
try:
    from hn import search_by_date as _hn_search  # Search HN stories and comments
                                                   # by keyword, date, and points.
    _HN = True
except ImportError:
    _HN = False

# --- Citation formatting (generates APA references from URLs) ---
# Check whether citationlib is installed without importing it.
# importlib.util.find_spec returns a spec object if the package exists,
# or None if it doesn't -- no side effects, no import needed.
_CITE = importlib.util.find_spec("citationlib") is not None


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION -- settings used across all modes
# ════════════════════════════════════════════════════════════════════════

# Where all intermediate work is saved. Every mode reads from and writes
# to this folder, so they can share data without passing it directly.
SCRATCHPAD = "scratchpad"

# The 9 languages we search in. Each gets its OWN native decomposition
# (not translated English -- the AI generates queries as a native speaker
# of that language would frame them).
#
# "why" explains why this language adds a perspective English misses.
# This isn't decoration -- developer culture is genuinely different in each.
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
# These are "LLM-speak" -- filler phrases that AI models overuse.
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
    # AI models produce filler in every language, not just English.
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
# SEARCH -- functions that find things on the internet
# ════════════════════════════════════════════════════════════════════════

def init_search(serper_key=None, brave_key=None):
    """
    Set up the search engine. Called once at the start of explore/deepen.
    Currently uses ddgs (free, no key needed).
    The serper_key/brave_key parameters are kept for future expansion
    but are not used with the current DDG-only setup.
    """
    pass  # ddgs needs no setup -- it just works


def search_web(query, n=5, region="us-en"):
    """
    Search the web for a query and return up to n results.
    region (e.g. "jp-jp", see LANGUAGES) favors results from that country.

    Each result is a dictionary with keys:
      - "title":   the page title
      - "url":     the full URL
      - "snippet": a preview of the page content
      - "domain":  just the domain name (e.g., "bbc.com")

    Two-tier search:
      1. ddgs (primary -- free, no API key, queries several engines)
      2. Google via multi_search_engine (fallback -- used when DDG returns
         empty, which happens with some non-English queries. Google's index
         is deeper for Japanese, Chinese, Korean content.)

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
            # Log the error instead of silently swallowing it.
            if attempt == 1:
                print(f"      Web search error: {type(e).__name__}: {e}")

    # ── Tier 2: Google fallback (when DDG returns empty) ────────────
    # DDG sometimes returns nothing for non-English queries or very
    # specific searches. Google's index is significantly deeper,
    # especially for Japanese, Chinese, Korean, and Russian content.
    # We only try Google if DDG returned zero results (not as a
    # supplement -- we don't want duplicate results).
    if not results and _SEARCH_GOOGLE:
        try:
            # _google_search is the search() function from multi_search.
            # search_engine="google" uses Google without an API key (scrapes HTML).
            # The return format is a list, but the exact structure varies --
            # we handle both dict-style and tuple-style results.
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

    # Drop results without a usable web address (some engines occasionally
    # return an empty or relative link -- there's nothing to fetch there)
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

    # ── Semantic Scholar (200M+ papers, great for CS/AI) ────────────
    if _S2:
        try:
            # Create a Semantic Scholar client and search
            for p in SemanticScholar().search_paper(query, limit=n):
                papers.append({
                    "title": p.title or "",
                    "domain": "semanticscholar.org",
                    # Build a URL to the paper's page
                    "url": p.url or f"https://www.semanticscholar.org/paper/{p.paperId}",
                    # Take first 500 characters of the abstract
                    "snippet": (p.abstract or "")[:500],
                    # getattr(p, "year", None) means: get p.year if it exists,
                    # otherwise return None (nothing). Safer than p.year which
                    # would crash if the paper doesn't have a year field.
                    "year": getattr(p, "year", None),
                    "citations": getattr(p, "citationCount", None),
                })
        except Exception as e:
            print(f"      Semantic Scholar error: {type(e).__name__}: {e}")

    # ── arXiv (preprints -- very recent, not yet peer-reviewed) ──────
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
                    "snippet": (p.summary or "")[:500],
                    # p.published is a datetime object; .year extracts just the year
                    "year": p.published.year if p.published else None,
                })
        except Exception as e:
            print(f"      arXiv error: {type(e).__name__}: {e}")

    return papers


def search_hn(query, n=10):
    """
    Search Hacker News for developer discussions about a topic.

    Hacker News is a forum where developers discuss tech. It's run by
    Y Combinator (a famous startup incubator). The discussions are often
    high-quality and represent what developers actually think, not what
    companies want them to think.

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
                    "snippet": (item.get("story_text") or "")[:300],
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
# CONTENT EXTRACTION -- reading the actual text from web pages
# ════════════════════════════════════════════════════════════════════════

def fetch_page(url, max_chars=5000):
    """
    Visit a web page and extract its main article text.

    This is a three-tier system:
      Tier 1: trafilatura (fast, no browser, works ~80% of the time)
      Tier 2: camoufox (stealth browser, beats bot detection, slower)
      Tier 3: give up and return None

    Returns the extracted text (up to max_chars), or None if it fails.

    Why two tiers? Most websites serve their content as plain HTML,
    which trafilatura can read directly. But some sites (Reddit, LinkedIn,
    news paywalls) detect automated access and block it. Camoufox opens
    a real browser that looks like a human, so these sites let it through.
    """

    # ── Tier 1: trafilatura (fast, no browser needed) ───────────────
    if _TRAF:
        try:
            # Step 1: Download the raw HTML of the page
            html = trafilatura.fetch_url(url)

            if html:
                # Step 2: Extract just the article text from all that HTML
                # include_tables=True: keep data tables (useful for research)
                # favor_recall=True: get more content even if some noise sneaks in
                text = trafilatura.extract(html, include_tables=True, favor_recall=True)

                # Only return it if we got something substantial (100+ chars)
                # A page that returns only 10 characters probably failed
                if text and len(text.strip()) > 100:
                    return text[:max_chars]  # Truncate to max length
        except Exception:
            pass  # trafilatura failed -- fall through to Tier 2

    # ── Tier 2: camoufox (stealth browser, for protected sites) ─────
    if _CAMO:
        try:
            # "with Camoufox(...) as browser" opens a stealth Firefox browser.
            # headless=True means invisible (no window pops up on your screen).
            # When the "with" block ends, the browser closes automatically.
            with Camoufox(headless=True) as browser:
                page = browser.new_page()       # Open a new tab
                page.goto(url, timeout=20000)   # Navigate to the URL (20 second timeout)

                # Wait until the page's HTML is loaded (some pages load slowly)
                page.wait_for_load_state("domcontentloaded")

                # Run JavaScript IN the page to extract its text.
                # This is like opening the browser console and typing code.
                # What the JavaScript does:
                #   1. Remove navigation, headers, footers, sidebars
                #   2. Find the main content area (<main>, <article>, etc.)
                #   3. Return its text content
                text = page.evaluate("""() => {
                    for (const s of ['nav','header','footer','aside','.sidebar'])
                        document.querySelectorAll(s).forEach(e => e.remove());
                    const m = document.querySelector('main,article,[role="main"]');
                    return (m || document.body).innerText;
                }""")

                if text and len(text.strip()) > 100:
                    # re.sub(r"\s+", " ", text) replaces multiple whitespace
                    # characters (spaces, tabs, newlines) with a single space.
                    # .strip() removes leading/trailing whitespace.
                    # NOTE: r"\s+" NOT r"\\s+" -- double backslash would match
                    # a literal backslash character, not whitespace.
                    return re.sub(r"\s+", " ", text).strip()[:max_chars]
        except Exception as e:
            print(f"      camoufox extraction error: {type(e).__name__}")

    # ── Tier 3: give up ─────────────────────────────────────────────
    return None  # Couldn't read the page with any method


# ════════════════════════════════════════════════════════════════════════
# AI -- talking to AI models via the OpenRouter API
# ════════════════════════════════════════════════════════════════════════

def call_ai(prompt, api_key, model, temp=0.6, system=""):
    """
    Send a prompt to an AI model and get back its response.

    This is the core of what makes the toolkit "AI-powered."
    It sends a message to a model (Claude, GPT, Llama, Mistral, etc.)
    via the OpenRouter API and returns what the model says.

    Parameters:
        prompt   -- what you want the AI to do (the question/instruction)
        api_key  -- your OpenRouter API key (the password)
        model    -- which AI model to use (e.g., "google/gemini-2.5-flash-lite")
        temp     -- "temperature" -- controls randomness. 0 = very predictable,
                   1 = very creative. 0.6 is a good middle ground for research.
        system   -- optional "system prompt" that tells the AI how to behave
                   (e.g., "you are an academic researcher, be precise")

    Returns the AI's response as a text string, or "" if something went wrong.

    HOW AN API CALL WORKS (ELI5):
    Your code sends a message over the internet to OpenRouter's server.
    OpenRouter forwards it to the AI model you chose.
    The AI thinks and generates a response.
    OpenRouter sends that response back to your code.
    The whole trip takes 2-30 seconds depending on how much you asked for.
    """

    # Build the messages list. AI APIs use a "conversation" format:
    # each message has a "role" (who's talking) and "content" (what they said).
    msgs = []

    # The "system" message sets the AI's behavior (optional)
    if system:
        msgs.append({"role": "system", "content": system})

    # The "user" message is your actual question/instruction
    msgs.append({"role": "user", "content": prompt})

    try:
        # Send the request to OpenRouter's API endpoint.
        # requests.post() sends an "HTTP POST request" -- the standard way
        # to send data to a web server.
        r = requests.post(
            # The URL of OpenRouter's chat completions endpoint
            "https://openrouter.ai/api/v1/chat/completions",

            # Headers tell the server who we are and what format we're using
            headers={
                # "Bearer" + API key is the standard authentication format
                "Authorization": f"Bearer {api_key}",
                # Tell the server we're sending JSON (structured data)
                "Content-Type": "application/json",
            },

            # The actual data we're sending (as JSON)
            json={
                "model": model,           # Which AI model to use
                "messages": msgs,         # The conversation so far
                "temperature": temp,      # How random/creative to be
            },

            timeout=180,  # Wait up to 3 minutes for a response (AI can be slow)
        )

        # If the server returned an error (400, 401, 500, etc.), crash here
        # so we go to the "except" block below. "raise_for_status" means
        # "if the response was an error, raise an exception."
        r.raise_for_status()

        # Parse the JSON response. The AI's text is nested inside:
        # response → choices → [first item] → message → content
        # It's structured this way because the API can return multiple
        # "choices" (alternative responses), but we just want the first one.
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        # Something went wrong (no internet, bad API key, model overloaded, etc.)
        print(f"   AI error: {e}")
        return ""  # Return empty string so the calling code doesn't crash


# ════════════════════════════════════════════════════════════════════════
# SCRATCHPAD -- saving and loading intermediate work
# ════════════════════════════════════════════════════════════════════════
#
# The scratchpad is a folder where every mode saves its output.
# Think of it like a shared notebook: explore writes its findings,
# analyze reads them and adds NLP results, stats reads those and
# adds statistical tests, synthesize reads everything and writes
# the thesis. Each mode reads from the previous mode's work.

def pad(name, content):
    """
    Save a piece of work to the scratchpad folder as a .md file.

    Parameters:
        name    -- filename (without .md extension)
        content -- the text to save

    Example: pad("01_definitions", "# Definitions\n\n...")
    Creates: scratchpad/01_definitions.md
    """
    # Create the scratchpad folder if it doesn't exist yet.
    # exist_ok=True means "don't crash if it already exists."
    Path(SCRATCHPAD).mkdir(exist_ok=True)

    # Build the full file path: scratchpad/01_definitions.md
    p = os.path.join(SCRATCHPAD, f"{name}.md")

    # Open the file for writing ("w") with UTF-8 encoding
    # (UTF-8 handles all languages -- Japanese, Chinese, Arabic, etc.)
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
# SOURCE CLASSIFICATION -- what kind of source is this?
# ════════════════════════════════════════════════════════════════════════
#
# Not all sources are equally trustworthy. An academic paper from arxiv.org
# is more reliable than a blog post on medium.com. This function looks at
# the domain name and assigns a quality label.
#
# These labels are used in analysis (do vendor sources have different
# sentiment than academic ones?) and in the report (label vendor-funded
# studies as such).

def classify_source(domain):
    """
    Given a domain name (like "arxiv.org" or "techcrunch.com"),
    return a quality label (like "academic" or "journalism").

    Uses pattern matching: if the domain contains "arxiv" anywhere,
    it's academic. If it contains "techcrunch", it's journalism.

    any(x in d for x in [...]) means:
    "is ANY of these strings found inside d?"
    It returns True if at least one match is found.
    """
    d = domain.lower()  # Lowercase so "ArXiv.org" and "arxiv.org" both match

    # Country-style endings (".edu", ".go.jp") are checked as SUFFIXES --
    # "edu" as a plain substring would also match e.g. "education-news.com".
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
        return "consulting"  # Note: consulting firms have their own biases

    # Companies that SELL AI coding tools. Plain github.com is NOT here:
    # most of it is user repositories, not GitHub's own publications.
    if any(x in d for x in ["github.blog", "docs.github.com", "microsoft.com", "openai.com",
                            "jetbrains", "stackoverflow.blog", "anthropic.com", "cursor.com",
                            "ibm.com", "tabnine", "codeium", "windsurf", "replit.com",
                            "sourcegraph", "kimi.ai", "aliyun.com"]):
        return "vendor/primary"  # These companies SELL the tools being studied

    if any(x in d for x in ["news.ycombinator", "dev.to", "lobste.rs", "reddit.com", "qiita.com",
                            "zenn.dev", "note.com", "tistory.com", "brunch.co.kr", "velog.io",
                            "habr.com"]):
        return "experiential/forum"  # Real developer discussions

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

    We only try if the text is at least 50 characters long --
    shorter text doesn't give the detector enough to work with.
    """
    if _LANG and text and len(text) > 50:
        try:
            # Only analyze first 500 chars (faster). langdetect returns
            # regional codes like "zh-cn" / "zh-tw"; keep just the base
            # code so it matches the keys in LANGUAGES ("zh").
            return _detect_lang(text[:500]).split("-")[0]
        except Exception:
            pass  # Detection failed -- return 'unknown'
    return "unknown"


# ════════════════════════════════════════════════════════════════════════
# CAPABILITIES -- show what's installed and working
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

    We match EVERY quoted string first (short ones included) and only then
    filter by length. Matching just long strings (r'"([^"]{10,})"') pairs
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
    quoted CLAIMS ("superficial", "cognitive load eroding,") and quoted
    source TITLES, which then got searched as if they were queries. Quoted
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
# BATCHING -- let the AI read more text than fits in one prompt
# ════════════════════════════════════════════════════════════════════════
#
# Used by ask_agent (answer from ALL research), analyze (topic clustering)
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
    IN BATCH ORDER. make_prompt(batch) builds each prompt. A failed call
    gives "" (call_ai already printed why). Lower workers if rate-limited.
    """
    from concurrent.futures import ThreadPoolExecutor  # Run calls in parallel

    def one(batch):
        return call_ai(make_prompt(batch), api_key, model, temp=temp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, batches))  # map keeps the original order


# ════════════════════════════════════════════════════════════════════════
# EVIDENCE CARDS -- one short English summary per source
# ════════════════════════════════════════════════════════════════════════
#
# A card is 1-2 sentences of the concrete findings, numbers and claims in a
# source, written in English whatever the source's language, tagged with
# the source number: "[44] Korean survey: Copilot raised PR completions 26%..."
#
# Why: the corpus is ~150 sources in 9 languages, ~400,000 characters.
# Cards shrink it ~10x while keeping every source, so:
#   - synthesize can show the AI EVERY source (not the first 10)
#   - analyze can cluster sources by THEME (clustering raw multilingual text
#     gives topics labelled "die / der / de")
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
                  s.get("content", "")[:3000]) for s in todo]
        answers = ai_map(
            make_batches(parts, batch_chars=24000),
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
