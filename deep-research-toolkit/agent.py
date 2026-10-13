#!/usr/bin/env python3
"""agent.py: command-line entry point of the Deep Research Toolkit.
Run: python agent.py MODE [--key KEY] ["question"]
Modes: explore, synthesize, deepen, analyze, stats, viz, ask, simulate, verify
Typical workflow: explore → analyze → stats → synthesize → deepen → analyze → viz → simulate
"""
import argparse
import faulthandler
import os
import subprocess
import sys

# Print every thread's stack on a C-level crash (segfault, double free)
faulthandler.enable()


def use_toolkit_venv():
    """Re-run this script with the .venv Python when install.py created one."""
    venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    python = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
              else os.path.join(venv, "bin", "python"))
    if os.path.exists(python) and os.path.realpath(sys.prefix) != os.path.realpath(venv):
        sys.exit(subprocess.call([python, os.path.abspath(__file__), *sys.argv[1:]]))


def main():
    ap = argparse.ArgumentParser(description="Deep Research Toolkit")
    ap.add_argument("mode", choices=[
        "explore", "synthesize", "deepen",
        "analyze", "stats", "viz", "ask", "simulate", "verify",
    ])
    ap.add_argument("--key", default=None)
    # Default model: cheap, 1M-token context, reliable in all 9 languages
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--serper", default=None)
    ap.add_argument("--brave", default=None)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--corpus", default="scratchpad")
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("question", nargs="?", default=None)
    a = ap.parse_args()

    modes = {
        "explore":    ("explore",     "run_explore"),
        "synthesize": ("synthesize",  "run_synthesize"),
        "deepen":     ("deepen",      "run_deepen"),
        "analyze":    ("analyze",     "run_analyze"),
        "stats":      ("stats",       "run_stats"),
        "viz":        ("viz",         "run_viz"),
        "ask":        ("ask_agent",   "run_ask"),
        "simulate":   ("simulate",    "run_simulate"),
        "verify":     ("verify",      "run_verify"),
    }
    mod_name, fn_name = modes[a.mode]
    module = __import__(mod_name)
    getattr(module, fn_name)(a)

if __name__ == "__main__":
    use_toolkit_venv()
    main()
