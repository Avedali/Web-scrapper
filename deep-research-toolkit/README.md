# Deep Research Toolkit

A multilingual, multi-phase Python research pipeline for the Sciences Po capstone: "Evaluate how coding assistants are changing the software development business."

## How It Works

1. **Explore** -- searches the web in 9 languages (English, French, German, Japanese, Korean, Chinese, Portuguese, Spanish, Russian), each with native-script queries. Gathers 60-100 sources, classifies their quality (academic, vendor, journalism, forum), and runs a red-team + bias audit.
2. **Analyze** -- runs Hugging Face NLP on the full corpus: multilingual sentiment scoring, zero-shot topic classification, named entity extraction, and BERTopic topic modeling. Chi-squared test checks whether sentiment is associated with source type.
3. **Stats** -- Mann-Whitney U (vendor vs non-vendor sentiment), Kruskal-Wallis (sentiment across languages), Spearman correlations, chi-squared with Fisher's exact fallback for sparse tables, and structural equation modeling (SEM) via semopy with proper dummy coding.
4. **Synthesize** -- reads ALL evidence and proposes 3 defensible theses ranked by evidence strength. You pick one. It generates a thesis-driven outline and writes each section grounded in cited sources. Voice cleaner strips LLM filler in 6 languages.
5. **Deepen** -- finds weak spots in the thesis and runs targeted searches to fill evidence gaps.
6. **Charts** -- generates every chart twice: paper version (captioned, labeled) and slide version (finding as title, one accent color, gray for non-focal).
7. **Simulate** -- gives AI models identical coding tasks (product search page, Kanban board, data explorer) at three skill levels (layman/junior/senior) and compares output quality. User chooses which models to benchmark.
8. **Ask** -- teammates ask questions about the research in plain language. 5-turn conversation memory.

## Output

- **Markdown report** with inline citations and APA 7th references
- **JSON metadata** (sources, quality breakdown, language breakdown)
- **Scratchpad** folder with all intermediate work (definitions, decomposition, red-team critiques, bias audit, raw synthesis, voice-cleaned output)
- **Charts** in paper and slide formats (PNG + interactive HTML)
- **Simulation HTML files** -- open in a browser, compare side by side

## Quick Start (for teammates)

1. Download [Thonny](https://thonny.org) (comes with Python)
2. Open `install.py` → click ▶ Run → wait → paste API key when asked
3. Open `launcher.py` → click ▶ Run → type a number

## Recommended Pipeline

```
explore → analyze → stats → synthesize → deepen → analyze → viz → simulate
```

Opinion comes AFTER data, not before.

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
| `requirements.txt` | All dependencies |

## Important Notes

- All search results are real -- no fallback to example data.
- The toolkit combines web-sourced information with AI analysis. The AI writes from the scratchpad only, never from its training data.
- Before submitting: run `launcher.py → option 9` to delete your API key from the `.env` file.
- Cost (measured on a real run, default model `google/gemini-2.5-flash-lite`): about $0.10 for a full pipeline run, plus 1-2 cents per Ask question.
