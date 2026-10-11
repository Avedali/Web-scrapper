#!/usr/bin/env python3
# ↑ This line tells your computer "run this file using Python 3."
#   You don't need to worry about it -- it's just a standard header.

"""
install.py -- Run this ONCE. It does everything.
=============================================
Open this file in Thonny and click the green Run button.
Or double-click it. Or type: python install.py

It will:
  1. Install all dependencies (takes 2-5 minutes)
  2. Ask for your OpenRouter API key and save it
  3. Download the Camoufox browser (if you want it)
  4. Test that everything works
  5. Tell you what to do next

(Why not "setup.py"? Python packaging tools treat any file with that name
as a package build script and may run it -- e.g. "pip install ." would
execute this installer and hang waiting for input.)
"""

# These are built-in Python tools we need. They come with Python, no install needed.
import subprocess  # Lets us run terminal commands from inside Python (like "pip install X")
import sys         # Gives us info about the Python version we're running
import os          # Lets us work with files and folders (create, read, check if they exist)
import importlib.util  # Lets us check if a package is installed without importing it


def main():
    """This is the main function. Everything happens inside here."""

    # Print a welcome banner so the user knows what's happening
    print("=" * 60)                          # Print a line of 60 "=" characters
    print("DEEP RESEARCH TOOLKIT -- SETUP")
    print("=" * 60)
    print()                                   # Empty line for readability
    print("   This will install everything you need.")
    print("   It takes about 2-5 minutes. Just wait.\n")  # \n = extra empty line

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: INSTALL ALL THE LIBRARIES (packages) THE TOOLKIT NEEDS
    # ════════════════════════════════════════════════════════════════════
    #
    # Python by itself is like an empty kitchen. "Packages" (also called
    # "libraries") are like appliances -- each one does a specific job.
    # We need to install about 25 of them. This step does that automatically.

    print("━" * 50)
    print("STEP 1: Installing Python packages...")
    print("━" * 50)
    print("   (This is the longest step. Be patient.)\n")

    # We organize the packages into groups so you can see progress.
    # Each group is a list of package names.

    # --- Group 1: Basic tools ---
    core = [
        "requests",    # Makes web requests (like opening a URL in a browser, but in code)
        "pandas",      # Works with data tables (like Excel but in Python)
        "langdetect",  # Detects what language a text is written in (English? French? Japanese?)
    ]

    # --- Group 2: Web searching and reading web pages ---
    search = [
        "duckduckgo-search",    # Searches DuckDuckGo (primary search engine)
        "multi_search_engine",  # Google fallback for non-English queries where DDG returns empty
        "trafilatura",          # Extracts the article text from a web page (ignores ads, menus, etc.)
    ]

    # --- Group 3: Searching academic papers and developer forums ---
    academic = [
        "semanticscholar",  # Searches 200 million+ academic papers
        "arxiv",            # Searches arXiv (where scientists post research papers)
        "python-hn",        # Searches Hacker News (a popular developer discussion site)
    ]

    # --- Group 4: AI/NLP tools (BIG downloads -- ~2-3 GB total) ---
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

    if nlp_choice == "n":
        # Light install: just TextBlob (tiny, fast, English-only but functional)
        nlp = [
            "textblob",  # Simple sentiment analysis (fallback)
        ]
        print("   → Light install selected (TextBlob only)")
    else:
        # Full install: HF transformers + PyTorch + BERTopic
        nlp = [
            "transformers",  # Hugging Face: pretrained AI models for text analysis
            "torch",         # PyTorch: the engine that runs those AI models (~2GB!)
            "bertopic",      # Neural topic modeling (discovers themes in texts)
        ]
        print("   → Full install selected (this will take a few minutes)")

    # --- Group 5: Statistics and chart-making ---
    stats_viz = [
        "scipy",        # Scientific computing: statistical tests (chi-squared, etc.)
        "pingouin",     # Statistics made easy: effect sizes, power analysis, Bayes Factor
        "researchpy",   # Research-formatted statistical output (ready for papers)
        "statsmodels",  # Advanced stats: logistic regression, time series
        "matplotlib",   # Makes charts and graphs (the classic Python chart library)
        "seaborn",      # Makes prettier statistical charts (built on matplotlib)
        "plotly",       # Makes interactive charts you can zoom/hover (for presentations)
        "kaleido",      # Lets plotly save charts as PNG images (for slides)
    ]

    # --- Group 6: Tools that analyze code quality ---
    code_analysis = [
        "bandit",  # Scans Python code for security vulnerabilities
        "radon",   # Measures code complexity (how tangled/simple the code is)
        "pylint",  # Grades code quality on a 0-10 scale
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
                # This is the key line: it runs "pip install <package>" in the background.
                # subprocess.check_call runs a terminal command from Python.
                # sys.executable is the path to the Python you're using right now.
                # "-m pip install -q" means "use pip to install quietly (less output noise)."
                # stdout=DEVNULL and stderr=DEVNULL hide pip's verbose output.
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    stdout=subprocess.DEVNULL,  # Hide normal output
                    stderr=subprocess.DEVNULL,  # Hide error output
                )
                print(f"      {pkg}")  # Success!

            except subprocess.CalledProcessError:
                # If the install fails (bad internet, incompatible package, etc.)
                # we note it but DON'T crash. The toolkit can work without most packages.
                failed.append(pkg)
                print(f"      {pkg} (failed -- will work without it)")

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
    # An API key is like a password that lets your code talk to an AI service.
    # OpenRouter is the service we use -- it gives access to many AI models
    # (Claude, GPT, Llama, Mistral, etc.) with one key.
    #
    # We save the key in a file called ".env" so you only type it once.
    # Every time the toolkit runs, it reads the key from that file.

    print("\n" + "━" * 50)
    print("STEP 2: API Key")
    print("━" * 50)

    # Figure out where to save the .env file (same folder as this script)
    # os.path.abspath(__file__) = the full path to THIS file (install.py)
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
    except (EOFError, KeyboardInterrupt):  # Handle Ctrl+C or weird input
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
        # os.environ is a dictionary of "environment variables" -- system-wide settings
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
        print("   Camoufox already installed!")
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
                # Download the actual browser binary (~300MB)
                subprocess.check_call(
                    [sys.executable, "-m", "camoufox", "fetch"])
                print("   Camoufox installed!")
            except Exception:
                print("   Camoufox install failed -- not a problem, toolkit works without it.")
        else:
            print("   Skipped.")

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
    # For example: you install "duckduckgo-search" but import "duckduckgo_search"
    tests = {
        "requests": "requests",
        "pandas": "pandas",
        "trafilatura": "trafilatura",
        "duckduckgo-search": "duckduckgo_search",    # Note: dash → underscore
        "multi_search_engine": "multi_engine_search",  # Note: install ≠ import name!
        "HF transformers": "transformers",
        "BERTopic": "bertopic",
        "pingouin": "pingouin",
        "plotly": "plotly",
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
    # DONE!
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
# This is the "entry point" -- the very first thing Python runs.
# It says: "if this file is being run directly (not imported by another
# file), call the main() function."
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
