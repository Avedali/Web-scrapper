Deep Research Toolkit

Multilingual research pipeline for the Sciences Po capstone: "Evaluate how coding assistants are changing the software development business."

How It Works:

Explore: Looks up the topic's key terms in dictionaries, then searches the web in 9 languages with a query prompt written in each language

Analyze: Scores sentiment, classifies topics and finds themes across all sources (Hugging Face models, BERTopic)

Stats: Tests whether sentiment differs by source type and language (Mann-Whitney, Kruskal-Wallis, chi-squared, SEM)

Synthesize: Writes a research brief from checked claims, each tied to the quote it comes from

Deepen: Finds thin themes in the brief and searches for more evidence

Viz: Draws charts of the findings, chosen from the shape of the data (data-to-viz.com), with APA sources under each chart

Simulate: Gives AI models the same coding tasks at three prompt skill levels and tests the pages they build

Ask: Answers teammates' questions about the research, with 5-turn memory

Verify: Runs the fact-check gate on a brief edited by hand

The brief contains:

Bottom Line
Key Findings (one per theme)
Findings by Theme
Where the Evidence Disagrees
Gaps in the Evidence
Business Stakes and Our View (for the audience set in AUDIENCE in synthesize.py)
Method
References (APA 7th)

Output:

Research Brief: Markdown report with numbered citations and a fact-check verdict (PASS or NEEDS REVIEW) at the top

Charts: Findings charts in charts_paper/ and charts_slide/, method charts in appendix/, as PNG and interactive HTML

Scratchpad: All intermediate work (definitions, queries, claim ledger, fact-check log, run log)

Simulation Pages: The HTML pages each model built, to open in a browser

Quick Start (step by step: START_HERE.txt):

1. Install Thonny (thonny.org), which comes with Python
2. Open install.py, click Run, paste the OpenRouter API key when asked
3. Open launcher.py, click Run, type a number, or R to run the whole pipeline unattended (about 2.5 hours)

From a terminal on a Mac: python3 install.py, then python3 launcher.py

Recommended order: explore → analyze → stats → synthesize → deepen → analyze → viz → simulate

Files:

START_HERE.txt: Step-by-step setup and first run on a Mac
install.py: Installs everything (run once)
launcher.py: Menu (run every time)
agent.py: Command line entry point (python3 agent.py explore, ...)
core.py: Shared tools (search, page reading, AI calls, APA references)
explore.py: Web research in 9 languages
analyze.py: NLP analysis
stats.py: Statistical tests
claims.py: Claim ledger (claims, their quotes, the checks)
synthesize.py: Research brief
verify.py: Fact-check gate
deepen.py: Evidence gap filling
viz.py: Charts
ask_agent.py: Teammate Q&A
simulate.py: Coding benchmark
requirements.txt: Package list

Important Notes:

Python 3.10 or newer is required; the Mac's built-in python3 (3.9) is too old
Homebrew's Python gets a private .venv, which the launcher uses automatically
Intel Macs need Python 3.12 or older for the full install (PyTorch dropped Intel Mac support after 2.2.2)
Keep the lid open during long runs; the toolkit keeps the Mac awake but closing the lid still sleeps it
All search results are real, with no fallback to example data
The brief is written only from checked claims, never from the AI's own knowledge
Every number in a chart is checked against its source
Run launcher option 9 before submitting to delete the API key from .env
Cost: about $0.40 per full run with the default model (google/gemini-2.5-flash-lite), plus 1-3 cents per Ask question
Time: explore 25-35 min, analyze about 1 hour on a 4-core laptop, synthesize 10-15 min
