#!/usr/bin/env python3

"""install.py: one-time setup for the toolkit.
Run: python install.py (or the Run button in Thonny). Not named setup.py, which pip would execute.
Steps: install packages, save the OpenRouter API key to .env, download Camoufox (optional) and browsers, test imports."""

import subprocess
import sys
import os
import importlib.util
import platform
import sysconfig
import tempfile



HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")

# Intel Macs: last PyTorch is 2.2.2, so dependent packages are pinned to compatible prebuilt versions
INTEL_MAC = sys.platform == "darwin" and platform.machine() == "x86_64"
INTEL_MAC_PINS = """torch==2.2.2
numpy<2
transformers<4.58
numba<=0.62.1
llvmlite<=0.45.1
"""


def _venv_python(venv_dir):
    """Return the path of the python program inside a venv."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _externally_managed():
    """True if this Python refuses "pip install" (Homebrew, some Linux system Pythons)."""
    in_venv = sys.prefix != sys.base_prefix
    marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
    return not in_venv and os.path.exists(marker)

def _camoufox_fetch():
    """Download the Camoufox browser (~300MB), first installing PyYAML with CLoader if missing."""
    try:
        from yaml import CLoader  # noqa: F401 (only checking it exists)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--ignore-installed", "pyyaml"])
    subprocess.check_call([sys.executable, "-m", "camoufox", "fetch"])
    # "fetch" can exit 0 after an error, so confirm the browser exists
    if subprocess.call(
            [sys.executable, "-c", "from camoufox.pkgman import installed_verstr; installed_verstr()"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        raise RuntimeError("the browser didn't download -- try again later with: "
                           "python -m camoufox fetch")


def main():
    """Run every setup step, from installing packages to the final test."""

    print("=" * 60)
    print("DEEP RESEARCH TOOLKIT -- SETUP")
    print("=" * 60)
    print()
    print("   This will install everything you need.")
    print("   It takes about 2-5 minutes. Just wait.\n")

    if sys.version_info < (3, 10):  # noqa: UP036 (users may run an old Python)
        print(f"   This is Python {sys.version.split()[0]} -- the toolkit needs 3.10 or newer.")
        print("   Easiest fix: install Thonny (https://thonny.org), which comes with a")
        print("   recent Python, and run install.py from Thonny. Or install Python")
        print("   from https://www.python.org/downloads/ and use that.")
        input("   Press Enter to close...")
        sys.exit(1)

    # Externally managed Python: create .venv and rerun this installer inside it
    if _externally_managed():
        print("   This Python doesn't allow installing packages into it (Homebrew or")
        print("   system Python), so the toolkit gets its own private copy in .venv ...")
        if not os.path.exists(_venv_python(VENV_DIR)):
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        sys.exit(subprocess.call([_venv_python(VENV_DIR), os.path.abspath(__file__)]))

    pip_extra = []
    if INTEL_MAC:
        pins = os.path.join(tempfile.gettempdir(), "deep_research_intel_mac_pins.txt")
        with open(pins, "w") as f:
            f.write(INTEL_MAC_PINS)
        pip_extra = ["-c", pins]
        print("   Intel Mac detected -- using package versions that still support it.\n")

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: INSTALL PACKAGES
    # ════════════════════════════════════════════════════════════════════

    print("━" * 50)
    print("STEP 1: Installing Python packages...")
    print("━" * 50)
    print("   (This is the longest step. Be patient.)\n")

    core = [
        "requests",    # HTTP downloads
        "pandas",      # Data tables
        "langdetect",  # Language detection
    ]

    search = [
        "ddgs",                 # Multi-engine web search (primary)
        "multi_search_engine",  # Google fallback for non-English queries
        "trafilatura",          # Article text extraction
        "pypdf",                # Text of PDF reports and papers
    ]

    academic = [
        "semanticscholar",  # Academic paper search
        "arxiv",            # arXiv search
        "python-hn",        # Hacker News search
    ]

    print("\n    The NLP tools (Hugging Face + PyTorch) are ~2-3 GB to download.")
    print("   They give you sentiment analysis, topic modeling, and NER.")
    print("   Without them, the toolkit falls back to simpler tools (TextBlob).")
    print("   Install full NLP? (y/n -- type 'n' if you're on slow WiFi)")
    try:
        nlp_choice = input("   > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        nlp_choice = "y"

    if INTEL_MAC and sys.version_info >= (3, 13) and nlp_choice != "n":
        # PyTorch 2.2.2 exists only for Python <= 3.12
        print("   On an Intel Mac, the NLP models need Python 3.12 or older (this is")
        print(f"   {sys.version.split()[0]}). Using the light install -- for the full one, install")
        print("   Python 3.12 from python.org and run install.py with it.")
        nlp_choice = "n"

    if nlp_choice == "n":
        nlp = [
            "textblob",  # English-only sentiment (fallback)
        ]
        print("   → Light install selected (TextBlob only)")
    else:
        nlp = [
            "transformers",  # Hugging Face pretrained models
            "torch",         # PyTorch, runs the models (~2GB)
            "sentencepiece", # Tokenizer for the multilingual models
            "protobuf",      # Needed by sentencepiece tokenizers
            "bertopic",      # Topic modeling
        ]
        print("   → Full install selected (this will take a few minutes)")

    stats_viz = [
        "scipy",        # Statistical tests
        "pingouin",     # Effect sizes, power analysis, Bayes Factor
        "researchpy",   # Paper-formatted stats output
        "statsmodels",  # Regression, time series
        "matplotlib",   # Static charts
        "seaborn",      # Statistical charts
        "plotly",       # Interactive charts
        "kaleido",      # Plotly PNG export
        "wordcloud",    # Word cloud of findings
    ]

    code_analysis = [
        "bandit",  # Security scan
        "radon",   # Code complexity
        "pylint",  # Code quality score
        "playwright",  # Browser check of simulated pages
    ]

    citations = [
        "citationlib",  # APA/MLA citations
    ]

    groups = [
        ("Core", core),
        ("Search & extraction", search),
        ("Academic & forums", academic),
        ("NLP (may take a few minutes)", nlp),
        ("Statistics & visualization", stats_viz),
        ("Code analysis", code_analysis),
        ("Citations", citations),
    ]

    failed = []

    for group_name, packages in groups:
        print(f"\n   {group_name}...")

        for pkg in packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", pkg, *pip_extra],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True, check=True,
                )
                print(f"      {pkg}")

            except subprocess.CalledProcessError as e:
                failed.append(pkg)
                print(f"      {pkg} (failed -- will work without it)")
                reasons = [line.strip() for line in (e.stderr or "").splitlines()
                           if line.strip() and not line.lower().startswith(("[notice]", "warning"))]
                if reasons:
                    print(f"         reason: {reasons[-1][:200]}")

    if failed:
        print(f"\n    {len(failed)} packages failed: {', '.join(failed)}")
        print("   The toolkit will still work -- those features will be skipped.")
    else:
        print("\n   All packages installed!")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: SAVE THE API KEY
    # ════════════════════════════════════════════════════════════════════

    print("\n" + "━" * 50)
    print("STEP 2: API Key")
    print("━" * 50)

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    existing_key = ""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    existing_key = line.split("=", 1)[1].strip()

    if existing_key:
        print(f"   Found saved key: {existing_key[:8]}...{existing_key[-4:]}")
        print("   Press Enter to keep it, or paste a new one:")
    else:
        print()
        print("   You need an OpenRouter API key to use AI features.")
        print("   Get one free at: https://openrouter.ai")
        print()
        print("   Paste your key here (starts with 'sk-or-'):")

    try:
        new_key = input("   > ").strip()
    except (EOFError, KeyboardInterrupt):
        new_key = ""

    if new_key:
        key = new_key
    elif existing_key:
        key = existing_key
    else:
        key = ""
        print("    No key provided. You can add it later in the .env file.")

    if key:
        with open(env_path, "w") as f:
            f.write(f"OPENROUTER_API_KEY={key}\n")
        print("   Key saved to .env")

        os.environ["OPENROUTER_API_KEY"] = key

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: BROWSERS (CAMOUFOX OPTIONAL)
    # ════════════════════════════════════════════════════════════════════

    print("\n" + "━" * 50)
    print("STEP 3: Camoufox (stealth browser -- optional)")
    print("━" * 50)

    if importlib.util.find_spec("camoufox"):
        print("   Camoufox package found -- making sure the browser is downloaded...")
        try:
            _camoufox_fetch()
            print("   Camoufox ready!")
        except Exception as e:
            print(f"   Browser download failed ({e}) -- not a problem, toolkit works without it.")
    else:
        print("   Camoufox is a stealth browser for hard-to-scrape sites.")
        print("   It's ~300MB and optional. Skip if unsure.")
        print("   Install? (y/n)")
        try:
            answer = input("   > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            print("   Installing camoufox (this takes a minute)...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "camoufox[geoip]"])
                _camoufox_fetch()
                print("   Camoufox installed!")
            except Exception as e:
                print(f"   Camoufox install failed ({e}) -- not a problem, toolkit works without it.")
        else:
            print("   Skipped.")

    # Chromium for simulate's page check (~150MB)
    if importlib.util.find_spec("playwright"):
        print("\n   Downloading the browser simulate uses to test pages (~150MB)...")
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                                  stdout=subprocess.DEVNULL)
            print("   Browser ready!")
        except Exception:
            print("   Browser download failed -- simulate will try Chrome/Edge instead.")

    # Chrome for kaleido PNG export, downloaded only if a test chart fails
    if importlib.util.find_spec("kaleido"):
        test_chart = ("import plotly.graph_objects as go, tempfile, os; "
                      "go.Figure().write_image(os.path.join(tempfile.gettempdir(), 'kaleido_test.png'))")
        if subprocess.call([sys.executable, "-c", test_chart],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            print("\n   Downloading Chrome for saving charts as images (~150MB)...")
            # certifi certificates: python.org Pythons on a Mac may have none
            env = dict(os.environ)
            try:
                import certifi
                env["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                pass
            try:
                subprocess.check_call([sys.executable, "-c", "import kaleido; kaleido.get_chrome_sync()"],
                                      stdout=subprocess.DEVNULL, env=env)
                print("   Chart export ready!")
            except Exception:
                print("   Download failed -- charts are still saved as interactive HTML.")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4: TEST IMPORTS
    # ════════════════════════════════════════════════════════════════════

    print("\n" + "━" * 50)
    print("STEP 4: Testing...")
    print("━" * 50)

    tests = {
        "requests": "requests",
        "pandas": "pandas",
        "trafilatura": "trafilatura",
        "pypdf": "pypdf",
        "ddgs": "ddgs",
        "multi_search_engine": "multi_search",  # Install name differs from import name
        "HF transformers": "transformers",
        "sentencepiece": "sentencepiece",
        "BERTopic": "bertopic",
        "pingouin": "pingouin",
        "plotly": "plotly",
        "wordcloud": "wordcloud",
        "bandit": "bandit",
        "langdetect": "langdetect",
        "semanticscholar": "semanticscholar",
    }

    working = 0

    for name, module in tests.items():
        try:
            __import__(module)
            print(f"   {name}")
            working += 1
        except ImportError:
            print(f"   {name} (not critical)")

    # ════════════════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print(f"SETUP COMPLETE -- {working}/{len(tests)} components working")
    print("=" * 60)
    print()
    print("   WHAT TO DO NEXT:")
    print()
    # Macs and Linux have only python3; Windows has python
    command = "python launcher.py" if os.name == "nt" else "python3 launcher.py"
    print(f"   In this window, type:  {command}")
    print("   (In Thonny: open launcher.py and click Run.)")
    print()
    print("   The launcher has a simple menu -- type R to run everything.")
    print()
    if not key:
        print("    Don't forget to add your API key to .env!")
        print()

    # Keep a double-clicked window open
    input("   Press Enter to close...")


if __name__ == "__main__":
    main()
