#!/usr/bin/env python3
"""
Deep Research Toolkit
======================
Usage:
    python agent.py explore    --key KEY    # 1.a  Exploratory research (no opinion, just gather)
    python agent.py synthesize              # 1.b  Read everything → propose thesis → outline
    python agent.py deepen     --key KEY    # 1.c  Targeted research to support the thesis
    python agent.py analyze                 # 2.a  NLP: sentiment, zero-shot, NER, topics
    python agent.py stats                   # 2.b  Statistical tests (pingouin)
    python agent.py viz                     # 3    Charts (paper + slide versions)
    python agent.py ask        --key KEY "q"# 4    Teammates ask plain-language questions
    python agent.py simulate   --key KEY    # 5    Visual benchmark (search page / kanban / data explorer)

Typical workflow:
    explore → analyze → stats → synthesize → deepen → analyze → viz → simulate
"""
import argparse
import faulthandler
import os
import subprocess
import sys

# If a C library crashes (no Python error, just "Segmentation fault" or
# "double free"), print where every thread was -- so the log says what broke
faulthandler.enable()


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


def main():
    ap = argparse.ArgumentParser(description="Deep Research Toolkit")
    ap.add_argument("mode", choices=[
        "explore", "synthesize", "deepen",
        "analyze", "stats", "viz", "ask", "simulate",
    ])
    ap.add_argument("--key", default=None)
    # Default model: cheap, 1M-token context, and reliable in all 9 languages.
    # (meta-llama/llama-4-scout was the old default; via OpenRouter it often
    # returned Chinese/Korean queries with the native characters missing.)
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
    }
    mod_name, fn_name = modes[a.mode]
    module = __import__(mod_name)
    getattr(module, fn_name)(a)

if __name__ == "__main__":
    use_toolkit_venv()   # Packages may live in .venv (see install.py)
    main()
