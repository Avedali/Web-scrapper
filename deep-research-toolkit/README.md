# Deep Research Toolkit

A multilingual, multi-phase Python research pipeline for the Sciences Po capstone: "Evaluate how coding assistants are changing the software development business."

## How It Works

1. **Explore** -- searches the web in 9 languages (English, French, German, Japanese, Korean, Chinese, Portuguese, Spanish, Russian), each with native-script queries. Gathers 60-100 sources, classifies their quality (academic, vendor, journalism, forum), and runs a red-team + bias audit.
2. **Analyze** -- runs Hugging Face NLP on the full corpus: multilingual sentiment scoring, zero-shot topic classification, named entity extraction, and BERTopic topic modeling. Chi-squared test checks whether sentiment is associated with source type.
3. **Stats** -- Mann-Whitney U (vendor vs non-vendor sentiment), Kruskal-Wallis (sentiment across languages), Spearman correlations, chi-squared with Fisher's exact fallback for sparse tables, and structural equation modeling (SEM) via semopy with proper dummy coding.
4. **Synthesize** -- reads ALL evidence and proposes 3 defensible theses ranked by evidence strength. You pick one. It generates a thesis-driven outline and writes each section grounded in cited sources. Voice cleaner strips LLM filler in 6 languages.
5. **Deepen** -- finds weak spots in the thesis and runs targeted searches to fill evidence gaps.
6. **Charts**: charts of the findings first: up to 3 per theme of the brief, drawn from the figures its sources report. Six forms: every source's figure for one measure (with how each was obtained: experiment, survey, vendor claim...), groups inside one study, the answers to one survey question, the parts of a whole, expected vs measured, and change over the years. The AI picks the figures; code checks every number against its source, checks that comparisons stay inside one study, and allows only numbers on the chart in the title. Then the simulation results. Charts about the research method (sources per theme, sentiment by source type, languages, source mix) go in an `appendix/` folder. Every chart comes in a paper version (caption, axis labels) and a slide version (the finding as the title).
7. **Simulate** -- gives AI models identical coding tasks (product search page, Kanban board, data explorer) at three skill levels (layman/junior/senior) and compares output quality. User chooses which models to benchmark.
8. **Ask** -- teammates ask questions about the research in plain language. 5-turn conversation memory.

## Output

- **Markdown report** with inline citations and APA 7th references
- **JSON metadata** (sources, quality breakdown, language breakdown)
- **Scratchpad** folder with all intermediate work (definitions, decomposition, red-team critiques, bias audit, raw synthesis, voice-cleaned output)
- **Charts** of the findings, plus method charts in `appendix/`, in paper and slide formats (PNG + interactive HTML)
- **Simulation HTML files** -- open in a browser, compare side by side

## Quick Start (for teammates)

1. Download [Thonny](https://thonny.org) (comes with Python)
2. Open `install.py` → click ▶ Run → wait → paste API key when asked
3. Open `launcher.py` → click ▶ Run → type a number, or **R** to run the
   whole pipeline unattended (~2.5 hours; the log is saved in `scratchpad/`)

### On a Mac

- **Thonny is the easiest route.** From Terminal instead, use `python3`
  (`python3 install.py`, then `python3 launcher.py`) -- Macs have no `python`.
- A Mac's built-in `python3` is 3.9, which is too old. The installer
  says so and points you to Thonny or python.org; use 3.10 or newer.
- Homebrew's Python doesn't allow installing packages into it. The
  installer then makes a private environment in `.venv`, and the launcher
  uses it automatically -- nothing to do.
- **Intel Macs:** PyTorch stopped supporting them after version 2.2.2, so
  the installer picks matching older package versions. That needs Python
  3.12 or older; on 3.13+ an Intel Mac gets the light install (no NLP models).
  Apple Silicon (M1-M4) Macs use the current versions on any Python.
- Long runs keep the Mac awake by themselves (`caffeinate`), but closing
  the lid still puts it to sleep -- keep it open while explore/analyze run.

## Recommended Pipeline

```
explore → analyze → stats → synthesize → deepen → analyze → viz → simulate
```

Opinion comes AFTER data, not before.

## How the brief stays tied to its sources

Synthesize doesn't write from memory, and nobody has to check it by hand:

1. **Claim ledger** (`claims.py`). Every usable source's claims are listed,
   each with the sentence of the source that states it. Code checks the
   quote is really in the source (a made-up quote drops the claim), and a
   local model checks the quote supports the claim *by meaning*. A claim
   that overstates its quote ("the sky is blue" from "often perceived as
   blue") is corrected to the quote's hedge and scope, not deleted. Reports
   of the same finding from several sources are merged, and each finding
   goes to one section. Cached in `scratchpad/claims.json`.
2. **Writing.** Each section is written from its own findings, citing them
   by ID; code turns the IDs into the right source numbers. There's no word
   quota: hard findings get explained, simple ones get a sentence, and a
   finding the writer doesn't use must be skipped with a reason
   (`scratchpad/synthesis_03b_coverage.md`).
3. **Fact-check gate** (`verify.py`). Rules in code (every number must be
   in the cited sources, at most 3 sources per claim, no repeats), the local
   judge, an AI review that corrects or removes what fails, then an
   independent audit of 20 random sentences against their sources' full
   text. The verdict -- **PASS** or **NEEDS REVIEW**, with the reasons -- is
   at the top of the brief, in `scratchpad/brief_verdict.json`, and in the
   "Run everything" summary. Details: `scratchpad/synthesis_05_fact_check.md`.

Edited a brief by hand? `python agent.py verify brief.md` (or launcher
option **V**) runs the gate on it and writes a checked copy and a report.

## Files

| File | What it does |
|---|---|
| `install.py` | Run once. Installs everything. |
| `launcher.py` | Menu interface. Run every time. |
| `agent.py` | CLI router (for terminal users) |
| `core.py` | Shared tools (search, AI, scratchpad) |
| `explore.py` | Exploratory research (9 languages) |
| `synthesize.py` | Thesis + outline from evidence |
| `deepen.py` | Targeted evidence gap filling |
| `analyze.py` | NLP analysis (HF multilingual models + BERTopic) |
| `stats.py` | Statistical tests (pingouin, semopy, scipy) |
| `viz.py` | Charts (plotly, dual paper/slide format) |
| `ask_agent.py` | Teammate Q&A (with conversation memory) |
| `simulate.py` | Visual coding benchmark |
| `claims.py` | Claim ledger: checked claims the brief is written from |
| `verify.py` | Fact-check gate (also `python agent.py verify brief.md`) |
| `requirements.txt` | All dependencies |

## Important Notes

- All search results are real -- no fallback to example data.
- The toolkit combines web-sourced information with AI analysis. The AI writes from the scratchpad only, never from its training data.
- Before submitting: run `launcher.py → option 9` to delete your API key from the `.env` file.
- Cost (default model `google/gemini-2.5-flash-lite`, ~290 sources, estimated from a real test run): about $0.40 for a full pipeline run, plus 1-3 cents per Ask question.
- Time: explore ~25-35 min; analyze ~1 hour on a 4-core laptop (it reads every source in full -- set `ANALYZE_CHARS` in analyze.py to trade depth for speed); synthesize ~10-15 min.
