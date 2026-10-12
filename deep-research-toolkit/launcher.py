#!/usr/bin/env python3
# ↑ Standard header: "run this with Python 3"

"""
launcher.py -- The friendly front door.
========================================
Open this in Thonny and click Run. Or double-click it.
Type a number. That's it.
"""

# Built-in Python tools (no install needed)
import os          # Work with files, folders, and environment variables
import sys         # Info about the current Python installation
import shutil      # Find programs on this computer (caffeinate on a Mac)
import subprocess  # Run other scripts/commands from inside this one



def use_toolkit_venv():
    """
    If install.py had to put the packages in a private environment (.venv --
    it does that for Homebrew's Python on a Mac), run this script with that
    environment's Python instead, so every package is found.
    """
    venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    python = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
              else os.path.join(venv, "bin", "python"))
    if os.path.exists(python) and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        sys.exit(subprocess.call([python, os.path.abspath(__file__), *sys.argv[1:]]))

def load_key():
    """
    Load the API key from the .env file that install.py created.

    An API key is like a password that lets our code talk to AI services.
    We saved it to a file called ".env" during setup so you don't have
    to type it every time.

    Returns the key as a text string, or "" if no key was found.
    """

    # Build the path to the .env file (same folder as this script)
    # os.path.abspath(__file__) = full path to THIS file (launcher.py)
    # os.path.dirname(...)     = just the folder part (without the filename)
    # os.path.join(..., ".env")= stick ".env" onto the end of that folder path
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    # Check if the .env file exists
    if os.path.exists(env_path):
        with open(env_path) as f:          # Open the file for reading
            for line in f:                 # Read it line by line
                # Look for the line that contains our key
                if line.startswith("OPENROUTER_API_KEY="):
                    # Split on the first "=" → take everything after it
                    # .strip() removes any trailing spaces or newline characters
                    key = line.split("=", 1)[1].strip()

                    # Also store it as an "environment variable" so other
                    # Python scripts in this session can find it automatically.
                    # Think of environment variables as system-wide sticky notes.
                    os.environ["OPENROUTER_API_KEY"] = key

                    return key  # Hand the key back to whoever called this function

    # If we get here, no .env file was found (or it didn't contain the key).
    # Try the environment variable directly as a last resort.
    # os.environ.get("X", "") means "get X, or return empty string if X doesn't exist"
    return os.environ.get("OPENROUTER_API_KEY", "")


def run_mode(mode, extra_args=None, log=None):
    """
    Run one of the toolkit's modes (explore, analyze, stats, etc.)

    Under the hood, this calls: python agent.py <mode>
    It's like typing that command in a terminal, but the launcher does it for you.
    The API key reaches the mode through the OPENROUTER_API_KEY environment
    variable (which child processes inherit), NOT through a --key argument:
    command-line arguments are visible to other programs and users on the
    machine (Task Manager, ps), environment variables are not.

    Parameters:
        mode       -- which mode to run (e.g., "explore", "analyze", "ask")
        extra_args -- any additional arguments to pass (e.g., a question for ask mode)
        log        -- an open file: run unattended and copy all output into it

    Returns the mode's exit code (0 = it finished without crashing).
    """

    # Build the command as a list of strings.
    # sys.executable = the path to the Python that's running THIS script
    # "agent.py"     = the router script that dispatches to the right mode
    # mode           = which mode to run (e.g., "explore")
    cmd = [sys.executable, "agent.py", mode]

    # Load the API key into os.environ (load_key does that for us);
    # the subprocess below inherits it.
    load_key()

    # Add any extra arguments (like a question for ask mode)
    if extra_args:
        cmd += extra_args  # Adds them to the end of the command

    # Show the user what we're about to run
    print(f"\n   Running: {' '.join(cmd)}\n")

    # Actually run the command.
    # subprocess.run() starts a new process and waits for it to finish.
    # cwd = "current working directory" -- we set it to the toolkit folder
    # so the scripts can find each other.
    here = os.path.dirname(os.path.abspath(__file__))

    # On a Mac, "caffeinate -i" keeps the computer awake while the mode runs:
    # explore and analyze take 30-60 minutes, and a Mac that goes to sleep
    # pauses them. (Closing the lid still sleeps it -- keep it open.)
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]

    if log is None:
        return subprocess.run(cmd, cwd=here, check=False).returncode

    # Unattended: no keyboard (every question takes its default answer), and
    # everything printed is also saved to the log file for later reading
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=here, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return proc.wait()


# The modes "Run everything" runs, in order. Simulate comes before the
# charts, which include its scores. Deepen is left out: what it adds only
# reaches the brief after another Analyze + Synthesize.
FULL_RUN = ["explore", "analyze", "stats", "synthesize", "simulate", "viz"]


def run_everything():
    """Run FULL_RUN one after another with no questions, stopping at a failure."""
    import time
    from datetime import datetime

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "scratchpad"), exist_ok=True)
    log_path = os.path.join(here, "scratchpad",
                            f"run_log_{datetime.now():%Y%m%d_%H%M}.txt")
    results = []
    with open(log_path, "w", encoding="utf-8") as log:
        for mode in FULL_RUN:
            started = time.time()
            banner = f"\n{'#' * 60}\n#  {mode.upper()}  ({datetime.now():%H:%M})\n{'#' * 60}\n"
            print(banner)
            log.write(banner)
            code = run_mode(mode, log=log)
            minutes = (time.time() - started) / 60
            results.append((mode, code, minutes))
            if code != 0:
                break   # Later steps need this one's output -- stop here

        summary = "\n" + "=" * 50 + "\n  RUN EVERYTHING -- SUMMARY\n" + "=" * 50 + "\n"
        for mode, code, minutes in results:
            summary += f"  {'OK    ' if code == 0 else 'FAILED'}  {mode:11s} {minutes:5.0f} min\n"
        for mode in FULL_RUN[len(results):]:
            summary += f"  skipped {mode}\n"
        summary += f"\n  Full log: {log_path}\n"
        if all(code == 0 for _, code, _ in results) and len(results) == len(FULL_RUN):
            summary += "  The brief is in this folder (brief_*.md); charts and simulations\n"
            summary += "  are in scratchpad/. Optional next: 5 (Deepen), then 2 and 4 again.\n"
        print(summary)
        log.write(summary)


def main():
    """
    The main menu loop.

    Shows options, waits for the user to type a number, runs the
    corresponding mode, then shows the menu again.

    "while True" means "keep doing this forever" -- we break out
    of the loop when the user types 0 (quit).
    """

    while True:
        # ── Print the menu ──────────────────────────────────────────
        print()
        print("=" * 50)
        print("  DEEP RESEARCH TOOLKIT")
        print("=" * 50)
        print()
        print("  RESEARCH")
        print("    1 → Explore      Gather sources (no opinion yet)")
        print("    2 → Analyze      Run NLP on what we found")
        print("    3 → Stats        Run statistical tests")
        print("    4 → Synthesize   Let data propose a thesis")
        print("    5 → Deepen       Fill gaps in the thesis")
        print()
        print("  OUTPUT")
        print("    6 → Charts       Generate paper + slide charts")
        print("    7 → Simulate     Visual benchmark (3 real dev tasks)")
        print()
        print("  TEAM")
        print("    8 → Ask          Ask a question about the research")
        print()
        print("    R → Run everything  1, 2, 3, 4, 7, 6 in a row, no questions (~2.5 hours)")
        print()
        print("    9 → Clean up     Delete API key from .env (run before submitting!)")
        print("    0 → Quit")
        print()
        print("  Cost (default model Gemini 2.5 Flash Lite, ~290 sources): about $0.40")
        print("     for a full run of options 1-7, plus ~1-3 cents per Ask question")
        print()

        # ── Wait for the user to type something ────────────────────
        try:
            # input() pauses the program and waits for the user to type.
            # Whatever they type gets stored in the "choice" variable.
            # .strip() removes accidental spaces before/after their input.
            choice = input("  What do you want to do? → ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError: the input stream ended unexpectedly
            # KeyboardInterrupt: the user pressed Ctrl+C (force quit)
            # In both cases, just exit the loop gracefully.
            break

        # ── Handle each choice ─────────────────────────────────────
        # Python checks each "if/elif" in order. As soon as one matches,
        # it runs that block and skips the rest.

        # --- Option 0: Quit ---
        if choice == "0" or choice.lower() in ("quit", "exit", "q"):
            # choice.lower() converts to lowercase so "Q" and "q" both work.
            # "in (...)" checks if the value is any of those options.
            print("\n  Bye! \n")
            break  # Exit the "while True" loop → program ends

        # --- Option 1: Explore (gather research, no opinion) ---
        elif choice == "1":
            print("\n  This searches the web in 9 languages and gathers sources.")
            print("  It takes about 25-35 minutes (72 queries, 9 languages). Let it run.\n")
            # Ask for confirmation before starting a long process
            confirm = input("  Start? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("explore")  # Calls: python agent.py explore --key ...

        # --- Option 2: Analyze (NLP on collected sources) ---
        elif choice == "2":
            print("\n  Running sentiment analysis, topic modeling, and NER")
            print("  (reads every source in full -- about an hour on a laptop for ~300 sources)")
            print("  on the sources we already collected.\n")
            run_mode("analyze")  # Calls: python agent.py analyze

        # --- Option 3: Stats (statistical tests) ---
        elif choice == "3":
            print("\n  Running statistical tests on the analysis results.\n")

            # Some statistical tests can run on external data (like a survey CSV).
            # Ask if they have one; if not, it tests the corpus data from analyze mode.
            print("  Do you have a survey CSV file to analyze? (y/n)")
            has_csv = input("  → ").strip().lower()

            if has_csv == "y":
                # They have a CSV -- ask for the file path
                csv_path = input("  Path to the CSV file: → ").strip()
                # Pass the file path as an extra argument: --data path/to/file.csv
                run_mode("stats", ["--data", csv_path])
            else:
                # No external data -- just run stats on what analyze produced
                run_mode("stats")

        # --- Option 4: Synthesize (thesis from evidence) ---
        elif choice == "4":
            print("\n  Reading ALL evidence and proposing a thesis.")
            print("  Only do this AFTER running Explore + Analyze + Stats.\n")
            confirm = input("  Ready? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("synthesize")

        # --- Option 5: Deepen (fill evidence gaps) ---
        elif choice == "5":
            print("\n  Finding targeted evidence for weak parts of the thesis.\n")
            run_mode("deepen")

        # --- Option 6: Charts (paper + slide versions) ---
        elif choice == "6":
            print("\n  Generating charts in two formats:")
            print("     - Paper: captioned, labeled (for the written paper)")
            print("     - Slides: minimal, finding as title (for the presentation)\n")
            run_mode("viz")

        # --- Option 7: Simulate (visual coding benchmark) ---
        elif choice == "7":
            print("\n  Running the simulation benchmark.")
            print("  Asks AI models to build a product search page, a kanban")
            print("  board, and a data explorer -- each at three skill levels.")
            print("  Outputs are HTML files you can open in a browser.\n")
            confirm = input("  Start? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("simulate")

        # --- Option 8: Ask (teammate Q&A) ---
        elif choice == "8":
            print("\n  Ask anything about the research.")
            print("  Type your question, or just chat.\n")

            question = input("  Your question: → ").strip()

            if question:
                # Single question mode: pass the question as an extra argument
                run_mode("ask", [question])
            else:
                # No question typed → open interactive mode where they
                # can ask multiple questions in a row
                run_mode("ask")

        # --- Option R: everything, unattended ---
        elif choice.lower() == "r":
            print("\n  Runs Explore, Analyze, Stats, Synthesize, Simulate and Charts")
            print("  one after another -- about 2.5 hours and $0.40 -- with no more")
            print("  questions (each step takes its default). You can walk away.")
            if sys.platform == "darwin":
                print("  Your Mac is kept awake while it runs; keep the lid open.")
            if not load_key():
                print("\n  No API key found -- run install.py first, or add it to .env")
                continue
            if input("\n  Start? (y/n) → ").strip().lower() == "y":
                run_everything()

        # --- Option 9: Clean up (delete API key before submission) ---
        elif choice == "9":
            print("\n  This removes your API key from the .env file so you")
            print("  don't accidentally submit it with your assignment.")
            print("  You can always re-run install.py to add a new key later.\n")
            confirm = input("  Delete your API key? (y/n) → ").strip().lower()
            if confirm == "y":
                env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                if os.path.exists(env_path):
                    # Read the file, remove the API key line, keep everything else.
                    # We drop the line entirely rather than writing a placeholder
                    # value: load_key() and install.py would read a placeholder as a
                    # real key and send it to the API.
                    with open(env_path) as f:
                        lines = f.readlines()
                    with open(env_path, "w") as f:
                        for line in lines:
                            if not line.startswith("OPENROUTER_API_KEY="):
                                f.write(line)
                        f.write("# OPENROUTER_API_KEY deleted before submission -- re-run install.py\n")
                    # Also forget it for the rest of this launcher session
                    os.environ.pop("OPENROUTER_API_KEY", None)
                    print("  API key removed from .env")
                    print("  Safe to submit your toolkit now.")
                else:
                    print("  No .env file found -- nothing to clean.")

        # --- Anything else: not a valid option ---
        else:
            print(f"\n  '{choice}' isn't an option. Type a number 0-9, or R.\n")

        # After any mode finishes, pause so the user can read the output
        # before the menu reappears and pushes it off screen.
        input("\n  Press Enter to go back to the menu...")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT -- what happens when you run this file
# ════════════════════════════════════════════════════════════════════════
#
# "if __name__ == '__main__'" means "if this file is being run directly."
# (As opposed to being imported by another file, which we don't do here.)

if __name__ == "__main__":
    use_toolkit_venv()   # Packages may live in .venv (see install.py)

    # Before showing the menu, check if setup has been done.
    # We do this by looking for the .env file -- if it doesn't exist,
    # install.py hasn't been run yet (because install.py creates .env).
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if not os.path.exists(env_path):
        # No .env file → setup hasn't been run
        print()
        print("   First time? Run install.py first!")
        print("  Open install.py in Thonny and click Run.\n")

        try:
            answer = input("  Or run setup now? (y/n) → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            # Run install.py as a subprocess (just like run_mode does).
            # This installs packages, asks for the API key, etc.
            subprocess.run(
                [sys.executable, "install.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                check=False,
            )

    # Now show the main menu
    main()
