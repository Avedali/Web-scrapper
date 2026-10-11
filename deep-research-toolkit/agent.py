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

def main():
    ap = argparse.ArgumentParser(description="Deep Research Toolkit")
    ap.add_argument("mode", choices=[
        "explore", "synthesize", "deepen",
        "analyze", "stats", "viz", "ask", "simulate",
    ])
    ap.add_argument("--key", default=None)
    ap.add_argument("--model", default="meta-llama/llama-4-scout")
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
    main()
