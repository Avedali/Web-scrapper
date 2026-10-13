#!/usr/bin/env python3

"""launcher.py: numbered menu that runs each toolkit mode through agent.py.
Run: open in Thonny and click Run (or double-click), then type a number.
"""

import os
import sys
import shutil
import subprocess



def use_toolkit_venv():
    """Re-run this script with the .venv Python if install.py created one."""
    venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    python = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
              else os.path.join(venv, "bin", "python"))
    if os.path.exists(python) and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        sys.exit(subprocess.call([python, os.path.abspath(__file__), *sys.argv[1:]]))

def load_key():
    """Return the API key from .env (also set in os.environ), else from the environment, else ""."""

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()

                    # Child processes inherit the key from os.environ
                    os.environ["OPENROUTER_API_KEY"] = key

                    return key

    return os.environ.get("OPENROUTER_API_KEY", "")


def run_mode(mode, extra_args=None, log=None):
    """Run `python agent.py <mode>` and return its exit code.
    With log (an open file): run unattended and copy all output into it.
    """

    cmd = [sys.executable, "agent.py", mode]

    # Key goes via the environment, not --key: arguments are visible in ps
    load_key()

    if extra_args:
        cmd += extra_args

    print(f"\n   Running: {' '.join(cmd)}\n")

    here = os.path.dirname(os.path.abspath(__file__))

    # Mac: caffeinate keeps the computer awake during long modes
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]

    if log is None:
        return subprocess.run(cmd, cwd=here, check=False).returncode

    # Unattended: no stdin (defaults answer every question), output teed to log
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=here, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return proc.wait()


# Modes for "Run everything", in order (viz charts include simulate scores)
FULL_RUN = ["explore", "analyze", "stats", "synthesize", "simulate", "viz"]


def _brief_verdict(scratchpad):
    """Return "" if brief_verdict.json says PASS, else the reason the brief needs review."""
    import json
    try:
        with open(os.path.join(scratchpad, "brief_verdict.json"), encoding="utf-8") as f:
            v = json.load(f)
    except (OSError, ValueError):
        return "the brief has no fact-check verdict -- check it by hand"
    if v.get("status") == "PASS":
        return ""
    return "brief NEEDS REVIEW: " + "; ".join(v.get("reasons", [])) + \
           " (see scratchpad/synthesis_05_fact_check.md)"


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
            note = ""
            if mode == "synthesize" and code == 0:
                note = _brief_verdict(os.path.join(here, "scratchpad"))
            results.append((mode, code, minutes, note))
            if code != 0:
                break   # Later steps need this one's output

        summary = "\n" + "=" * 50 + "\n  RUN EVERYTHING -- SUMMARY\n" + "=" * 50 + "\n"
        for mode, code, minutes, note in results:
            status = "FAILED" if code else ("WARN  " if note else "OK    ")
            summary += f"  {status}  {mode:11s} {minutes:5.0f} min\n"
            if note:
                summary += f"          {note}\n"
        for mode in FULL_RUN[len(results):]:
            summary += f"  skipped {mode}\n"
        summary += f"\n  Full log: {log_path}\n"
        if all(code == 0 for _, code, _, _ in results) and len(results) == len(FULL_RUN):
            summary += "  The brief is in this folder (brief_*.md); charts and simulations\n"
            summary += "  are in scratchpad/. Optional next: 5 (Deepen), then 2 and 4 again.\n"
        print(summary)
        log.write(summary)


def main():
    """Show the menu and run the chosen mode until the user quits."""

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
        print("    4 → Synthesize   Write the research brief")
        print("    5 → Deepen       Find evidence for the brief's gaps")
        print()
        print("  OUTPUT")
        print("    6 → Charts       Generate paper + slide charts")
        print("    7 → Simulate     Visual benchmark (3 real dev tasks)")
        print()
        print("  TEAM")
        print("    8 → Ask          Ask a question about the research")
        print("    V → Verify       Fact-check a brief (e.g. after editing it by hand)")
        print()
        print("    R → Run everything  1, 2, 3, 4, 7, 6 in a row, no questions (~2.5 hours)")
        print()
        print("    9 → Clean up     Delete API key from .env (run before submitting!)")
        print("    0 → Quit")
        print()
        print("  Cost (default model Gemini 2.5 Flash Lite, ~290 sources): about $0.40")
        print("     for a full run of options 1-7, plus ~1-3 cents per Ask question")
        print()

        try:
            choice = input("  What do you want to do? → ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        # ── Handle each choice ─────────────────────────────────────
        if choice == "0" or choice.lower() in ("quit", "exit", "q"):
            print("\n  Bye! \n")
            break

        # --- 1: Explore ---
        elif choice == "1":
            print("\n  This searches the web in 9 languages and gathers sources.")
            print("  It takes about 25-35 minutes (72 queries, 9 languages). Let it run.\n")
            confirm = input("  Start? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("explore")

        # --- 2: Analyze ---
        elif choice == "2":
            print("\n  Running sentiment analysis, topic modeling, and NER")
            print("  (reads every source in full -- about an hour on a laptop for ~300 sources)")
            print("  on the sources we already collected.\n")
            run_mode("analyze")

        # --- 3: Stats, optionally on a survey CSV ---
        elif choice == "3":
            print("\n  Running statistical tests on the analysis results.\n")

            print("  Do you have a survey CSV file to analyze? (y/n)")
            has_csv = input("  → ").strip().lower()

            if has_csv == "y":
                csv_path = input("  Path to the CSV file: → ").strip()
                run_mode("stats", ["--data", csv_path])
            else:
                run_mode("stats")

        # --- 4: Synthesize ---
        elif choice == "4":
            print("\n  Reading all the evidence and writing the research brief.")
            print("  Only do this AFTER running Explore + Analyze + Stats.\n")
            confirm = input("  Ready? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("synthesize")

        # --- 5: Deepen ---
        elif choice == "5":
            print("\n  Searching for evidence on the gaps the brief lists.\n")
            run_mode("deepen")

        # --- 6: Charts ---
        elif choice == "6":
            print("\n  Generating charts in two formats:")
            print("     - Paper: captioned, labeled (for the written paper)")
            print("     - Slides: minimal, finding as title (for the presentation)\n")
            run_mode("viz")

        # --- 7: Simulate ---
        elif choice == "7":
            print("\n  Running the simulation benchmark.")
            print("  Asks AI models to build a product search page, a kanban")
            print("  board, and a data explorer -- each at three skill levels.")
            print("  Outputs are HTML files you can open in a browser.\n")
            confirm = input("  Start? (y/n) → ").strip().lower()
            if confirm == "y":
                run_mode("simulate")

        # --- 8: Ask (one question, or interactive if empty) ---
        elif choice == "8":
            print("\n  Ask anything about the research.")
            print("  Type your question, or just chat.\n")

            question = input("  Your question: → ").strip()

            if question:
                run_mode("ask", [question])
            else:
                run_mode("ask")

        # --- V: Verify ---
        elif choice.lower() == "v":
            print("\n  Checks every claim in a brief against its sources, fixes or")
            print("  removes what isn't supported, and writes a checked copy.")
            path = input("  Brief to check (Enter = the newest one): → ").strip()
            run_mode("verify", [path] if path else None)

        # --- R: Run everything ---
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

        # --- 9: Clean up ---
        elif choice == "9":
            print("\n  This removes your API key from the .env file so you")
            print("  don't accidentally submit it with your assignment.")
            print("  You can always re-run install.py to add a new key later.\n")
            confirm = input("  Delete your API key? (y/n) → ").strip().lower()
            if confirm == "y":
                env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                if os.path.exists(env_path):
                    # Drop the key line: a placeholder value would be read as a real key
                    with open(env_path) as f:
                        lines = f.readlines()
                    with open(env_path, "w") as f:
                        for line in lines:
                            if not line.startswith("OPENROUTER_API_KEY="):
                                f.write(line)
                        f.write("# OPENROUTER_API_KEY deleted before submission -- re-run install.py\n")
                    os.environ.pop("OPENROUTER_API_KEY", None)
                    print("  API key removed from .env")
                    print("  Safe to submit your toolkit now.")
                else:
                    print("  No .env file found -- nothing to clean.")

        else:
            print(f"\n  '{choice}' isn't an option. Type a number 0-9, R or V.\n")

        # Pause so the output stays on screen
        input("\n  Press Enter to go back to the menu...")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    use_toolkit_venv()

    # No .env: install.py has not been run
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if not os.path.exists(env_path):
        print()
        print("   First time? Run install.py first!")
        print("  Open install.py in Thonny and click Run.\n")

        try:
            answer = input("  Or run setup now? (y/n) → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            subprocess.run(
                [sys.executable, "install.py"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                check=False,
            )

    main()
