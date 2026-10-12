#!/usr/bin/env python3
# ↑ This line tells your computer to run this file with Python 3.
#   It is a standard header and needs no changes.

"""
install.py: run this once to set up the toolkit.
==================================================
Open this file in Thonny and click the green Run button.
Or double-click it. Or type: python install.py

It will:
  1. Install all dependencies (takes 2-5 minutes)
  2. Ask for your OpenRouter API key and save it
  3. Download the Camoufox browser (if you want it)
  4. Test that everything works
  5. Tell you what to do next

(The file is not called "setup.py" because Python packaging tools treat
any file with that name as a package build script and may run it. For
example, "pip install ." would execute this installer and hang waiting
for input.)
"""

# These are built-in Python tools we need. They come with Python, no install needed.
import subprocess  # Lets us run terminal commands from inside Python (like "pip install X")
import sys         # Gives us info about the Python version we're running
import os          # Lets us work with files and folders (create, read, check if they exist)
import importlib.util  # Lets us check if a package is installed without importing it
import platform      # Which kind of computer this is (Intel or Apple Silicon Mac)
import sysconfig     # Where this Python keeps its files
import tempfile      # A place for a temporary settings file



HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")

# Intel Macs: PyTorch stopped making Intel Mac versions after 2.2.2, so the
# packages around it must be versions that still work with it (and that
# still come ready-built for Intel Macs, so nothing needs compiling).
INTEL_MAC = sys.platform == "darwin" and platform.machine() == "x86_64"
INTEL_MAC_PINS = """torch==2.2.2
numpy<2
transformers<4.58
numba<=0.62.1
llvmlite<=0.45.1
"""


def _venv_python(venv_dir):
    """Path of the python program inside a virtual environment."""
    if os.name == "nt":   # Windows
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _externally_managed():
    """
    True if this Python refuses "pip install" (Homebrew's Python on a Mac,
    and some Linux system Pythons, mark themselves this way). Packages then
    have to go into a private environment (a "venv") instead.
    """
    in_venv = sys.prefix != sys.base_prefix
    marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
    return not in_venv and os.path.exists(marker)

def _camoufox_fetch():
    """
    Download the camoufox browser (~300MB). camoufox needs PyYAML's fast C
    loader, which some Linux/Mac Pythons ship without ("cannot import name
    'CLoader'"). So this first installs pip's own copy of PyYAML when it's missing.
    """
    try:
        from yaml import CLoader  # noqa: F401 (only checking it exists)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--ignore-installed", "pyyaml"])
    subprocess.check_call([sys.executable, "-m", "camoufox", "fetch"])
    # "fetch" can report an error and still exit normally, so check that the
    # browser is there before saying it's ready
    if subprocess.call(
            [sys.executable, "-c", "from camoufox.pkgman import installed_verstr; installed_verstr()"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        raise RuntimeError("the browser didn't download -- try again later with: "
                           "python -m camoufox fetch")


def main():
    """Runs every setup step, from installing packages to the final test."""

    # Print a welcome banner so the user knows what's happening
    print("=" * 60)                          # Print a line of 60 "=" characters
    print("DEEP RESEARCH TOOLKIT -- SETUP")
    print("=" * 60)
    print()                                   # Empty line for readability
    print("   This will install everything you need.")
    print("   It takes about 2-5 minutes. Just wait.\n")  # \n = extra empty line

    # The toolkit (and the current versions of pandas, numpy and PyTorch)
    # need Python 3.10 or newer. A Mac's built-in "python3" is 3.9.
    if sys.version_info < (3, 10):  # noqa: UP036 (users may run an old Python)
        print(f"   This is Python {sys.version.split()[0]} -- the toolkit needs 3.10 or newer.")
        print("   Easiest fix: install Thonny (https://thonny.org), which comes with a")
        print("   recent Python, and run install.py from Thonny. Or install Python")
        print("   from https://www.python.org/downloads/ and use that.")
        input("   Press Enter to close...")
        sys.exit(1)

    # Homebrew's Python (the usual "python3" on a developer's Mac) refuses to
    # install packages into itself. Make a private environment in .venv and
    # run this installer again inside it. The launcher uses .venv from then on.
    if _externally_managed():
        print("   This Python doesn't allow installing packages into it (Homebrew or")
        print("   system Python), so the toolkit gets its own private copy in .venv ...")
        if not os.path.exists(_venv_python(VENV_DIR)):
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        sys.exit(subprocess.call([_venv_python(VENV_DIR), os.path.abspath(__file__)]))

    # Pin package versions that still work on Intel Macs (see INTEL_MAC_PINS)
    pip_extra = []
    if INTEL_MAC:
        pins = os.path.join(tempfile.gettempdir(), "deep_research_intel_mac_pins.txt")
        with open(pins, "w") as f:
            f.write(INTEL_MAC_PINS)
        pip_extra = ["-c", pins]
        print("   Intel Mac detected -- using package versions that still support it.\n")

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: INSTALL ALL THE LIBRARIES (packages) THE TOOLKIT NEEDS
    # ════════════════════════════════════════════════════════════════════
    #
    # "Packages" (also called "libraries") are code that Python doesn't
    # include by default, such as pandas for data tables. Each one does a
    # specific job. This step installs about 25 of them automatically.

    print("━" * 50)
    print("STEP 1: Installing Python packages...")
    print("━" * 50)
    print("   (This is the longest step. Be patient.)\n")

    # We organize the packages into groups so you can see progress.
    # Each group is a list of package names.

    # --- Group 1: Basic tools ---
    core = [
        "requests",    # Downloads web pages from code (the same request a browser sends for a URL)
        "pandas",      # Works with data tables (rows and columns, as in Excel)
        "langdetect",  # Detects what language a text is written in (e.g. English, French, Japanese)
    ]

    # --- Group 2: Web searching and reading web pages ---
    search = [
        "ddgs",                 # Multi-engine web search (primary search engine)
        "multi_search_engine",  # Google fallback for non-English queries where DDG returns empty
        "trafilatura",          # Extracts the article text from a web page (ignores ads, menus, etc.)
    ]

    # --- Group 3: Searching academic papers and developer forums ---
    academic = [
        "semanticscholar",  # Searches 200 million+ academic papers
        "arxiv",            # Searches arXiv (where scientists post research papers)
        "python-hn",        # Searches Hacker News (a popular developer discussion site)
    ]

    # --- Group 4: AI/NLP tools (large downloads: ~2-3 GB total) ---
    # PyTorch alone is ~2GB. On slow WiFi this can take 30+ minutes.
    # We ask the user if they want the full install or a light version.
    print("\n    The NLP tools (Hugging Face + PyTorch) are ~2-3 GB to download.")
    print("   They give you sentiment analysis, topic modeling, and NER.")
    print("   Without them, the toolkit falls back to simpler tools (TextBlob).")
    print("   Install full NLP? (y/n -- type 'n' if you're on slow WiFi)")
    try:
        nlp_choice = input("   > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        nlp_choice = "y"

    if INTEL_MAC and sys.version_info >= (3, 13) and nlp_choice != "n":
        # PyTorch's last Intel Mac version (2.2.2) only exists for Python <= 3.12
        print("   On an Intel Mac, the NLP models need Python 3.12 or older (this is")
        print(f"   {sys.version.split()[0]}). Using the light install -- for the full one, install")
        print("   Python 3.12 from python.org and run install.py with it.")
        nlp_choice = "n"

    if nlp_choice == "n":
        # Light install: TextBlob only (small and fast; works for English only)
        nlp = [
            "textblob",  # Simple sentiment analysis (fallback)
        ]
        print("   → Light install selected (TextBlob only)")
    else:
        # Full install: HF transformers + PyTorch + BERTopic
        nlp = [
            "transformers",  # Hugging Face: pretrained AI models for text analysis
            "torch",         # PyTorch: the library that runs those AI models (~2GB)
            "sentencepiece", # Tokenizer the multilingual zero-shot + NER models need
            "protobuf",      # Needed by sentencepiece-based tokenizers
            "bertopic",      # Neural topic modeling (discovers themes in texts)
        ]
        print("   → Full install selected (this will take a few minutes)")

    # --- Group 5: Statistics and chart-making ---
    stats_viz = [
        "scipy",        # Scientific computing: statistical tests (chi-squared, etc.)
        "pingouin",     # Ready-made statistics: effect sizes, power analysis, Bayes Factor
        "researchpy",   # Research-formatted statistical output (ready for papers)
        "statsmodels",  # Advanced stats: logistic regression, time series
        "matplotlib",   # Makes charts and graphs (the classic Python chart library)
        "seaborn",      # Makes prettier statistical charts (built on matplotlib)
        "plotly",       # Makes interactive charts you can zoom/hover (for presentations)
        "kaleido",      # Lets plotly save charts as PNG images (for slides)
        "wordcloud",    # Draws the word cloud of the brief's findings
    ]

    # --- Group 6: Tools that analyze code quality ---
    code_analysis = [
        "bandit",  # Scans Python code for security vulnerabilities
        "radon",   # Measures code complexity (how tangled/simple the code is)
        "pylint",  # Grades code quality on a 0-10 scale
        "playwright",  # Opens the simulated pages in a browser to check they work
    ]

    # --- Group 7: Citation formatting ---
    citations = [
        "citationlib",  # Generates APA/MLA citations from URLs and DOIs
    ]

    # Put all groups into one big list with labels
    groups = [
        ("Core", core),
        ("Search & extraction", search),
        ("Academic & forums", academic),
        ("NLP (may take a few minutes)", nlp),
        ("Statistics & visualization", stats_viz),
        ("Code analysis", code_analysis),
        ("Citations", citations),
    ]

    # This list will track any packages that fail to install
    failed = []

    # Loop through each group, then each package in that group
    for group_name, packages in groups:
        print(f"\n   {group_name}...")  # Show which group we're installing

        for pkg in packages:
            try:
                # This line runs "pip install <package>" in the background.
                # subprocess.check_call runs a terminal command from Python.
                # sys.executable is the path to the Python you're using right now.
                # "-m pip install -q" means "use pip to install quietly (less output noise)."
                # stdout=DEVNULL and stderr=DEVNULL hide pip's verbose output.
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", pkg, *pip_extra],
                    stdout=subprocess.DEVNULL,  # Hide normal output
                    stderr=subprocess.PIPE,     # Keep errors, to show why it failed
                    text=True, check=True,
                )
                print(f"      {pkg}")  # Installed

            except subprocess.CalledProcessError as e:
                # If the install fails (bad internet, incompatible package, etc.)
                # the package goes on the failed list and the installer keeps going.
                # The toolkit can work without most packages.
                failed.append(pkg)
                print(f"      {pkg} (failed -- will work without it)")
                # pip's last error line says why (skipping its "notice"/warning lines)
                reasons = [line.strip() for line in (e.stderr or "").splitlines()
                           if line.strip() and not line.lower().startswith(("[notice]", "warning"))]
                if reasons:
                    print(f"         reason: {reasons[-1][:200]}")

    # Report results
    if failed:
        print(f"\n    {len(failed)} packages failed: {', '.join(failed)}")
        print("   The toolkit will still work -- those features will be skipped.")
    else:
        print("\n   All packages installed!")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: SAVE YOUR API KEY
    # ════════════════════════════════════════════════════════════════════
    #
    # An API key is a secret string that identifies you to an AI service and
    # lets your code send it requests. The toolkit uses OpenRouter, which gives
    # access to many AI models (Claude, GPT, Llama, Mistral, etc.) with one key.
    #
    # We save the key in a file called ".env" so you only type it once.
    # Every time the toolkit runs, it reads the key from that file.

    print("\n" + "━" * 50)
    print("STEP 2: API Key")
    print("━" * 50)

    # Figure out where to save the .env file (same folder as this script)
    # os.path.abspath(__file__) = the full path to this file (install.py)
    # os.path.dirname(...) = the folder containing this file
    # os.path.join(..., ".env") = add ".env" to that folder path
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    # Check if a key was already saved from a previous run
    existing_key = ""
    if os.path.exists(env_path):           # Does the .env file exist?
        with open(env_path) as f:          # Open it for reading
            for line in f:                 # Read each line
                if line.startswith("OPENROUTER_API_KEY="):  # Found the key line?
                    # Split on the first "=" and take everything after it
                    existing_key = line.split("=", 1)[1].strip()

    # Show appropriate message based on whether a key already exists
    if existing_key:
        # Show first 8 and last 4 characters (hide the middle for security)
        print(f"   Found saved key: {existing_key[:8]}...{existing_key[-4:]}")
        print("   Press Enter to keep it, or paste a new one:")
    else:
        print()
        print("   You need an OpenRouter API key to use AI features.")
        print("   Get one free at: https://openrouter.ai")
        print()
        print("   Paste your key here (starts with 'sk-or-'):")

    # Wait for the user to type or paste something
    try:
        new_key = input("   > ").strip()   # .strip() removes extra spaces/newlines
    except (EOFError, KeyboardInterrupt):  # Handle Ctrl+C or a closed input
        new_key = ""

    # Decide which key to use
    if new_key:                   # User typed a new key
        key = new_key
    elif existing_key:            # User pressed Enter, keep old key
        key = existing_key
    else:                         # No key at all
        key = ""
        print("    No key provided. You can add it later in the .env file.")

    # Save the key to the .env file
    if key:
        with open(env_path, "w") as f:                  # Open .env for writing
            f.write(f"OPENROUTER_API_KEY={key}\n")       # Write the key
        print("   Key saved to .env")

        # Also set it for the current session (so testing works right away)
        # os.environ is a dictionary of "environment variables" (system-wide settings)
        os.environ["OPENROUTER_API_KEY"] = key

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: CAMOUFOX (OPTIONAL)
    # ════════════════════════════════════════════════════════════════════
    #
    # Camoufox is a stealth web browser that can visit websites that block
    # automated access (like Reddit, LinkedIn, some news sites).
    # It's optional because most sources the toolkit needs don't block us.
    # It's ~300MB so we ask before downloading it.

    print("\n" + "━" * 50)
    print("STEP 3: Camoufox (stealth browser -- optional)")
    print("━" * 50)

    # Check if Camoufox is already installed without importing it.
    # importlib.util.find_spec() looks for a package without loading it.
    # If it finds the package, it returns info about it. If not, it returns None.
    if importlib.util.find_spec("camoufox"):
        # The Python package alone isn't enough. The browser itself is a
        # separate ~300MB download (e.g. when camoufox came in through
        # requirements.txt). "fetch" does nothing if the browser is already there.
        print("   Camoufox package found -- making sure the browser is downloaded...")
        try:
            _camoufox_fetch()
            print("   Camoufox ready!")
        except Exception as e:
            print(f"   Browser download failed ({e}) -- not a problem, toolkit works without it.")
    else:
        # None means the package isn't installed
        print("   Camoufox is a stealth browser for hard-to-scrape sites.")
        print("   It's ~300MB and optional. Skip if unsure.")
        print("   Install? (y/n)")
        try:
            answer = input("   > ").strip().lower()  # .lower() so "Y" and "y" both work
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            print("   Installing camoufox (this takes a minute)...")
            try:
                # Install the package
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "camoufox[geoip]"])
                # Download the browser program itself (~300MB)
                _camoufox_fetch()
                print("   Camoufox installed!")
            except Exception as e:
                print(f"   Camoufox install failed ({e}) -- not a problem, toolkit works without it.")
        else:
            print("   Skipped.")

    # ── Browser for simulate's page check (~150MB) ──────────────────────
    # simulate opens every AI-generated page in an invisible browser to see
    # whether it works. Playwright needs its own copy of Chromium
    # (it can also use Google Chrome or Microsoft Edge if you have them).
    if importlib.util.find_spec("playwright"):
        print("\n   Downloading the browser simulate uses to test pages (~150MB)...")
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                                  stdout=subprocess.DEVNULL)
            print("   Browser ready!")
        except Exception:
            print("   Browser download failed -- simulate will try Chrome/Edge instead.")

    # ── Chrome for saving charts as PNG images ──────────────────────────
    # kaleido (plotly's image export) draws charts with Google Chrome. If
    # Chrome isn't installed, it can download its own copy (~150MB).
    if importlib.util.find_spec("kaleido"):
        test_chart = ("import plotly.graph_objects as go, tempfile, os; "
                      "go.Figure().write_image(os.path.join(tempfile.gettempdir(), 'kaleido_test.png'))")
        if subprocess.call([sys.executable, "-c", test_chart],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            print("\n   Downloading Chrome for saving charts as images (~150MB)...")
            # This download uses Python's own HTTPS, which on a Mac with
            # Python from python.org has no certificates until "Install
            # Certificates.command" is run. So this uses certifi's certificates.
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
    # STEP 4: TEST THAT EVERYTHING WORKS
    # ════════════════════════════════════════════════════════════════════
    #
    # We try to "import" each key package. Importing means loading it into
    # Python's memory. If the import works, the package is correctly installed.

    print("\n" + "━" * 50)
    print("STEP 4: Testing...")
    print("━" * 50)

    # Dictionary mapping friendly names to the actual Python module names
    # (sometimes the package name you install differs from the name you import)
    # For example: you install "multi_search_engine" but import "multi_search"
    tests = {
        "requests": "requests",
        "pandas": "pandas",
        "trafilatura": "trafilatura",
        "ddgs": "ddgs",
        "multi_search_engine": "multi_search",  # Install name ≠ import name
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

    working = 0  # Count how many passed

    for name, module in tests.items():
        try:
            __import__(module)  # __import__ is a built-in function that loads a module by name
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
    print("   Open launcher.py in Thonny and click Run.")
    print("   Or type: python launcher.py")
    print()
    print("   The launcher has a simple menu -- just type a number.")
    print()
    if not key:
        print("    Don't forget to add your API key to .env!")
        print()

    # Wait for the user to press Enter before closing the window.
    # Without this, if they double-clicked the file, the window would
    # close instantly and they'd never see the results.
    input("   Press Enter to close...")


# ════════════════════════════════════════════════════════════════════════
# This is the "entry point": the code that starts the program.
# It means: "if this file is being run directly (not imported by another
# file), call the main() function."
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
