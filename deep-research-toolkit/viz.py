"""
viz.py: Mode 3, draws every chart in a paper and a slide version.
Run: python agent.py viz  (or launcher.py, option 6)

How it works:
  Paper version: caption, axis labels, APA references (charts_paper/)
  Slide version: finding as title, big font, one accent color (charts_slide/)
  Findings charts: AI extracts tables of figures per theme; _form picks the chart
  Appendix charts: sources, sentiment, languages and topics of the corpus (appendix/)
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json
import sys
import os
import re
import datetime

from core import (call_ai, source_label, apa_reference, apa_in_text,
                  load_title_translations)
import verify



# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_viz(args):
    """Generate all charts from the analysis and simulation results."""

    print("=" * 60)
    print("MODE 3: VISUALIZE (paper + slide formats)")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("pip install plotly")

    paper_root = os.path.join(args.corpus, "charts_paper")
    slide_root = os.path.join(args.corpus, "charts_slide")
    os.makedirs(paper_root, exist_ok=True)
    os.makedirs(slide_root, exist_ok=True)

    csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
    if not os.path.exists(csv_path):
        sys.exit("Run 'python agent.py analyze' first.")

    df = pd.read_csv(csv_path)
    if "quality" in df.columns:
        df["source_type"] = df["quality"].map(source_label)
    charts = 0

    # ── Palette: one accent color for the main point ────────────────
    ACCENT = "#E63946"     # Red: the main point
    MUTED = "#A8DADC"      # Soft blue: context
    GRAY = "#B0B0B0"       # Gray: everything else
    DARK = "#1D3557"       # Navy: text and strong lines

    # ── Save in both formats ────────────────────────────────────────
    numbers = {"findings": 0, "appendix": 0}   # Next figure number in each group

    def save(fig, slug, caption, title_slide, appendix=False, keep_legend=False, note="", paper_note=""):
        """Save one chart as paper and slide versions; appendix charts are numbered A1, A2...
        note / paper_note: small print under the slide / paper chart (paper falls back to note)."""
        nonlocal charts
        paper_note = paper_note or note

        def room(text):
            """Height in pixels of a note under the chart (16 per line)."""
            return 16 * (text.count("<br>") + 1) + 14 if text else 0

        bottom = (100 if keep_legend else 20) + room(paper_note)
        if note:
            fig.add_annotation(text=paper_note, showarrow=False, xref="paper", yref="paper", align="left",
                               x=0, y=0, yshift=-bottom - 52, xanchor="left", yanchor="bottom",
                               font=dict(size=11, color="#6B6B6B"))
        group = "appendix" if appendix else "findings"
        numbers[group] += 1
        number = f"A{numbers[group]}" if appendix else str(numbers[group])
        name = f"{number}_{slug}" if appendix else f"{numbers[group]:02d}_{slug}"
        title_paper = f"Figure {number}: {caption}"
        paper_dir = os.path.join(paper_root, "appendix") if appendix else paper_root
        slide_dir = os.path.join(slide_root, "appendix") if appendix else slide_root
        os.makedirs(paper_dir, exist_ok=True)
        os.makedirs(slide_dir, exist_ok=True)

        # ── Paper version ───────────────────────────────────────────
        fig.update_layout(
            title=title_paper,
            title_font_size=14,
            font_size=11,
            template="plotly_white",
        )
        if note:
            fig.update_layout(margin=dict(b=bottom + 60))
        # PNG needs kaleido; without it, save HTML and warn once
        try:
            fig.write_image(
                os.path.join(paper_dir, f"{name}.png"),
                scale=2,
                width=900,
                height=550 + room(paper_note),
            )
        except Exception as e:
            fig.write_html(os.path.join(paper_dir, f"{name}.html"))
            if not hasattr(save, "_warned"):
                print(f"   PNG export failed -- saving HTML instead ({type(e).__name__})")
                if "chrome" in str(e).lower():
                    print("      Install Google Chrome, or run: plotly_get_chrome")
                else:
                    print("      pip install kaleido  (for static PNG export)")
                save._warned = True

        # ── Slide version ───────────────────────────────────────────
        if note:
            bottom = (100 if keep_legend else 20) + room(note)
            fig.layout.annotations[-1].update(text=note, yshift=-bottom - 52)
        fig.update_layout(
            title=title_slide,
            title_font_size=22,
            title_x=0.02,
            font_size=14,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            showlegend=keep_legend,
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
            margin=dict(l=20, r=20, t=90, b=bottom + (60 if note else 0)),
        )
        try:
            fig.write_image(
                os.path.join(slide_dir, f"{name}.png"),
                scale=2, width=1200, height=700 + room(note),
            )
        except Exception:
            pass  # Already warned above; the HTML file below is still saved

        fig.write_html(os.path.join(slide_dir, f"{name}.html"))

        charts += 1
        print(f"   {'appendix/' if appendix else ''}{name}")

    # ════════════════════════════════════════════════════════════════
    # FINDINGS CHARTS (from the brief)
    # ════════════════════════════════════════════════════════════════
    brief = None
    brief_path = os.path.join(args.corpus, "brief_findings.json")
    if not os.path.exists(brief_path):
        print("   No brief yet (run synthesize first): skipping the findings charts")
    else:
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)
        with open(os.path.join(args.corpus, "corpus.json"), encoding="utf-8") as f:
            by_idx = {int(src["idx"]): src for src in json.load(f)}
        load_title_translations(list(by_idx.values()), args.corpus)

        # ── Up to 3 charts per theme, form chosen by _form ──────────
        api_key = getattr(args, "key", None) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("   No API key: skipping the findings charts (they need one AI call per theme)")
        else:
            colors = {"accent": ACCENT, "dark": DARK, "muted": MUTED, "gray": GRAY, "blue": "#457B9D"}
            for c in _theme_charts(brief, by_idx, api_key, args.model):
                fig, legend = _draw(c, go, colors, datetime.date.today().year)
                srcs = [by_idx[i] for i in dict.fromkeys(p["idx"] for p in c["points"]) if i in by_idx]
                title = c["title"] or c["measure"]
                save(fig, re.sub(r"\W+", "_", f"{c['theme']} {c['form']}".lower()).strip("_")[:50],
                     f"{c['theme']}: {title}", _wrap(title, 70), keep_legend=legend,
                     note=_wrap(f"Sources: {apa_in_text(srcs)}", 150),
                     paper_note="Sources:<br>" + "<br>".join(
                         _wrap(re.sub(r"\*([^*]+)\*", r"<i>\1</i>", apa_reference(x)), 160) for x in srcs))

        # ── Word cloud of the findings ──────────────────────────────
        try:
            from wordcloud import WordCloud, STOPWORDS
        except ImportError:
            WordCloud = None
            print("   pip install wordcloud  (for the word cloud of the findings)")
        text = " ".join(f["claim"] for f in brief.get("findings", []))
        if WordCloud and text:
            import numpy as np
            # Leave out the question's words and words with the same stem
            stems = {_stem(w) for w in re.findall(r"[a-z]{3,}", brief.get("question", "").lower())}
            words = set(re.findall(r"[a-z]+", text.lower()))
            skip = (set(STOPWORDS) | WORDCLOUD_SKIP | {w for w in words if _stem(w) in stems}
                    | set(re.findall(r"[a-z]+", brief.get("question", "").lower())))
            text = " ".join(w for w in re.findall(r"[A-Za-z][\w-]*", text) if w.lower() not in skip)
            cloud = WordCloud(width=1800, height=900, background_color="white", stopwords=skip,
                              collocations=True, max_words=80, min_word_length=3, random_state=1,
                              color_func=lambda *a, **k: [DARK, "#457B9D", "#7FA7C9"][k["random_state"].randint(0, 2)])
            cloud.generate(re.sub(r"\d+([.,]\d+)?%?", " ", text))
            top = list(cloud.words_)[:3]
            fig = go.Figure(go.Image(z=np.array(cloud.to_image())))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(hovermode=False)
            save(fig, "word_cloud", "The most frequent words in the brief's findings",
                 f"The findings talk most about {', '.join(top[:-1])} and {top[-1]}" if len(top) > 1
                 else "The most frequent words in the findings")

    # ════════════════════════════════════════════════════════════════
    # APPENDIX: how the research was done
    # ════════════════════════════════════════════════════════════════

    # ── Sources cited per theme, by source type ─────────────────
    themes = [t for t in (brief or {}).get("themes", []) if t.get("sources_by_type")]
    if themes:
        kinds = sorted({k for t in themes for k in t["sources_by_type"]},
                       key=lambda k: -sum(t["sources_by_type"].get(k, 0) for t in themes))
        fixed = {"Research papers": DARK, "Other websites": MUTED}
        others = iter(["#457B9D", "#F4A261", "#2A9D8F", "#8D99AE", "#E9C46A", "#6D597A", GRAY])
        palette = {k: (ACCENT if k.startswith("AI tool makers") else fixed.get(k) or next(others, GRAY))
                   for k in kinds}
        fig = go.Figure()
        names = [t["title"] for t in themes]
        for k in kinds:
            fig.add_trace(go.Bar(
                y=names, x=[t["sources_by_type"].get(k, 0) for t in themes], name=k,
                orientation="h", marker_color=palette[k]))
        fig.update_layout(barmode="stack", yaxis_autorange="reversed")
        fig.update_xaxes(title="Sources cited")
        save(fig, "sources_per_theme",
             "Sources cited for each theme of the brief, by type of source",
             "Sources cited for each theme of the brief", appendix=True, keep_legend=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 1: Sentiment by source type (box plot)
    # ════════════════════════════════════════════════════════════════

    if "sentiment" in df.columns and "source_type" in df.columns:

        order = (
            df.groupby("source_type")["sentiment"]
            .median()
            .sort_values()
            .index
            .tolist()
        )

        fig = px.box(
            df, x="sentiment", y="source_type",
            category_orders={"source_type": order},
            labels={"source_type": "Type of source", "sentiment": "Sentiment (-1 negative to +1 positive)"},
            color_discrete_sequence=[MUTED],
            points="all",
        )

        fig.add_vline(x=0, line_dash="dash", line_color=GRAY, line_width=1)

        n_total = len(df)

        is_vendor = df["quality"].str.contains("vendor", case=False, na=False)
        if is_vendor.any() and (~is_vendor).any():
            v_mean = df.loc[is_vendor, "sentiment"].mean()
            o_mean = df.loc[~is_vendor, "sentiment"].mean()
            slide_title = (f"Sources from AI tool makers average {v_mean:+.2f} sentiment.<br>"
                           f"Other sources average {o_mean:+.2f}")
        else:
            slide_title = "Sentiment by source type"

        save(fig, "sentiment_by_source_type",
             f"Sentiment of each source, by type of source (N={n_total})",
             slide_title, appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 2: Sentiment by language (lollipop chart)
    # ════════════════════════════════════════════════════════════════

    if "sentiment" in df.columns and "lang" in df.columns:

        data = df.groupby("lang").agg(
            m=("sentiment", "mean"),
            n=("sentiment", "count"),
        ).reset_index()

        data = data[data["n"] >= 2].sort_values("m")

        if len(data) >= 2:

            fig = go.Figure()

            for _, row in data.iterrows():
                color = ACCENT if abs(row["m"]) > 0.1 else GRAY

                fig.add_trace(go.Scatter(
                    x=[0, row["m"]],
                    y=[row["lang"], row["lang"]],
                    mode="lines",
                    line=dict(color=GRAY, width=2),
                    showlegend=False,
                ))

                fig.add_trace(go.Scatter(
                    x=[row["m"]],
                    y=[row["lang"]],
                    mode="markers+text",
                    marker=dict(size=14, color=color),
                    text=[f"{row['m']:+.2f} (n={int(row['n'])})"],
                    textposition="middle right",
                    showlegend=False,
                ))

            fig.add_vline(x=0, line_dash="solid", line_color=DARK, line_width=1)

            lo_x, hi_x = min(0, data["m"].min()), max(0, data["m"].max())
            span = (hi_x - lo_x) or 1
            fig.update_xaxes(range=[lo_x - 0.05 * span, hi_x + 0.3 * span])

            lo, hi = data.iloc[0], data.iloc[-1]
            save(fig, "sentiment_by_language",
                 "Average sentiment of the sources in each language",
                 f"Most positive: {hi['lang']} ({hi['m']:+.2f}). "
                 f"Least positive: {lo['lang']} ({lo['m']:+.2f})", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 3: Language coverage (treemap)
    # ════════════════════════════════════════════════════════════════

    if "lang" in df.columns:

        counts = df["lang"].value_counts().reset_index()
        counts.columns = ["lang", "count"]

        fig = px.treemap(
            counts,
            path=["lang"],
            values="count",
            color="count",
            color_continuous_scale=["#F1FAEE", DARK],
        )

        save(fig, "language_coverage",
             f"Number of sources in each language (N={len(df)})",
             f"Sources in {df['lang'].nunique()} languages", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 4: Topic × Language (heatmap)
    # ════════════════════════════════════════════════════════════════

    # BERTopic themes from analyze, else zero-shot topics
    topic_col = "theme" if "theme" in df.columns else "topic_zs"

    if topic_col in df.columns and "lang" in df.columns:

        sub = df[df[topic_col].notna() & (df[topic_col] != "no usable evidence")]
        ct = pd.crosstab(sub["lang"], sub[topic_col])

        if ct.shape[0] >= 2 and ct.shape[1] >= 2:

            ct.columns = [
                c[:30] + "..." if len(c) > 30 else c
                for c in ct.columns
            ]

            fig = px.imshow(
                ct,
                text_auto=True,
                color_continuous_scale="YlOrRd",
                aspect="auto",
            )

            save(fig, "themes_by_language",
                 "Themes discussed by the sources in each language",
                 "Which languages discuss which themes", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 5: Source quality breakdown (lollipop)
    # ════════════════════════════════════════════════════════════════

    if "source_type" in df.columns:

        counts = df["source_type"].value_counts().sort_values()

        fig = go.Figure()

        for q, c in counts.items():
            color = ACCENT if q.startswith(("AI tool makers", "Low-quality")) else MUTED

            fig.add_trace(go.Scatter(
                x=[0, c], y=[q, q],
                mode="lines", line=dict(color=GRAY, width=2),
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=[c], y=[q],
                mode="markers+text",
                marker=dict(size=12, color=color),
                text=[str(c)],
                textposition="middle right",
                showlegend=False,
            ))

        fig.update_xaxes(range=[0, counts.max() * 1.15])
        save(fig, "source_types",
             f"Number of sources of each type (N={len(df)})",
             f"Where the {len(df)} sources come from", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHARTS 6-7: SIMULATION RESULTS (if simulate ran)
    # ════════════════════════════════════════════════════════════════

    sim_path = os.path.join(args.corpus, "simulation_results.json")

    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim = json.load(f)

        # ── Chart 6: Prompt skill level vs result ───────────────────
        vr = sim.get("visual_results", sim.get("coding_results", []))

        if vr:
            sim_df = pd.DataFrame(vr)

            level_col = "level" if "level" in sim_df.columns else "prompt_type"

            if level_col in sim_df.columns and "pro_met" in sim_df.columns:

                sim_df = sim_df[sim_df["pro_total"].notna() & (sim_df["pro_total"] > 0)].copy()
                sim_df["spec_pct"] = 100 * sim_df["pro_met"] / sim_df["pro_total"]
                fig = px.box(
                    sim_df, x=level_col, y="spec_pct",
                    color=level_col,
                    color_discrete_map={"layman": GRAY, "junior": MUTED, "senior": ACCENT},
                    category_orders={level_col: ["layman", "junior", "senior"]},
                    points="all",
                    labels={"spec_pct": "% of a professional's requirements met"},
                )
                fig.update_yaxes(range=[0, 105])

                means = sim_df.groupby(level_col)["spec_pct"].mean()
                slide_title = (f"Prompts written like a senior developer met {means.get('senior', 0):.0f}% "
                               f"of a professional's requirements.<br>A layman's prompts met "
                               f"{means.get('layman', 0):.0f}%")
                save(fig, "prompt_skill_vs_result",
                     "Share of a professional's requirements each AI-built page met, by how "
                     "skilled the prompt was",
                     slide_title)

            elif level_col in sim_df.columns:

                # No browser test results: code length by skill level
                fig = px.box(
                    sim_df, x=level_col, y="lines",
                    color=level_col,
                    color_discrete_map={
                        "layman": GRAY,
                        "junior": MUTED,
                        "senior": ACCENT,
                    },
                    category_orders={level_col: ["layman", "junior", "senior"]},
                    points="all",
                )

                med = sim_df.groupby(level_col)["lines"].median()
                if med.get("layman", 0) > 0 and "senior" in med:
                    ratio = med["senior"] / med["layman"]
                    slide_title = f"Senior prompts get {ratio:.1f}× more code than layman prompts"
                else:
                    slide_title = "Code output length by prompt skill level"

                save(fig, "prompt_skill_vs_result",
                     "Length of the code the AI wrote, by how skilled the prompt was",
                     slide_title)

        # ── Chart 7: Guardrail behavior ─────────────────────────────
        gr = sim.get("guardrail_results", [])

        if gr:
            gr_df = pd.DataFrame(gr)

            grade_map = {"SECURE": 3, "PARTIAL": 2, "INSECURE": 1}
            gr_df["score"] = gr_df["grade"].map(grade_map)

            fig = px.bar(
                gr_df, x="model", y="score",
                color="task_name",
                barmode="group",
                text="grade",
                color_discrete_sequence=[ACCENT, MUTED, DARK],
            )

            fig.update_yaxes(
                tickvals=[1, 2, 3],
                ticktext=["INSECURE", "PARTIAL", "SECURE"],
            )

            save(fig, "security_tests",
                 "How each AI model handled 3 requests for insecure code",
                 "Not all AI models warn about insecure code", keep_legend=True)

    # ── BERTopic charts (saved by analyze) ──────────────────────────
    bt_dir = os.path.join(args.corpus, "charts")
    existing_bt = (
        [f for f in os.listdir(bt_dir) if f.startswith("bertopic")]
        if os.path.exists(bt_dir) else []
    )
    if existing_bt:
        print(f"   BERTopic charts already in {bt_dir}/: {', '.join(existing_bt)}")

    print(f"\n{'='*60}")
    print(f"{charts} charts generated in two formats:")
    print(f"   Paper:  {paper_root}/ (captioned, axis labels)")
    print(f"   Slides: {slide_root}/ (minimal, finding as title)")
    print("   Charts about the research method are in the appendix/ subfolders.")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# ONE CHART PER KEY FINDING
# ════════════════════════════════════════════════════════════════════════

# Word cloud stop words besides common English: words every finding uses to report a result
WORDCLOUD_SKIP = {"according", "study", "studies", "survey", "surveys", "report", "reports", "reported",
                  "found", "finds", "shows", "showed", "percent", "respondents", "research", "year",
                  "years", "said", "says", "also", "may", "can", "will", "one", "two", "use", "used",
                  "using", "new", "many", "much", "including", "based", "per", "approximately",
                  "must", "making", "make", "entire", "least", "already", "without", "real", "leading",
                  "help", "result", "results", "modern", "understand", "increased", "increase", "improve",
                  "suggesting", "tool", "tools", "system", "systems", "within", "across", "among",
                  "well", "like", "need", "needs", "often", "way", "ways", "even", "still", "however"}
# Short words of other Latin-script languages, for claims claims.py left untranslated
WORDCLOUD_SKIP |= {"der", "die", "das", "und", "mit", "von", "für", "den", "dem", "des", "ist", "sind",
                   "les", "des", "une", "pour", "dans", "avec", "sur", "est", "sont", "par", "qui",
                   "los", "las", "una", "para", "con", "por", "que", "del", "como", "más",
                   "uma", "para", "com", "mais", "não", "dos", "das", "pelo", "pela"}

MAX_NUMBERED = 80     # Findings with numbers shown to the AI for one theme
MAX_POINTS = 20       # Figures in one table
TABLES_PER_THEME = 3  # Tables of figures the AI may extract for one theme
DISTRIBUTION_MIN = 8  # Figures of one measure that make a box plot
MAX_LINES = 5         # Lines in one line chart

# How a figure was obtained (one marker each, in chart order) and what kind of number it is
METHODS = ["experiment", "observational data", "survey", "vendor claim", "forecast", "not stated"]
STATISTICS = ["average", "median", "up to", "share", "change", "not stated"]

# Country names to ISO 3166 codes, for maps
COUNTRIES = {
    "united states": "USA", "usa": "USA", "u.s.": "USA", "america": "USA", "canada": "CAN",
    "mexico": "MEX", "brazil": "BRA", "argentina": "ARG", "chile": "CHL", "colombia": "COL",
    "united kingdom": "GBR", "uk": "GBR", "britain": "GBR", "ireland": "IRL", "france": "FRA",
    "germany": "DEU", "netherlands": "NLD", "belgium": "BEL", "switzerland": "CHE", "austria": "AUT",
    "spain": "ESP", "portugal": "PRT", "italy": "ITA", "poland": "POL", "sweden": "SWE",
    "norway": "NOR", "denmark": "DNK", "finland": "FIN", "russia": "RUS", "ukraine": "UKR",
    "turkey": "TUR", "israel": "ISR", "saudi arabia": "SAU", "united arab emirates": "ARE",
    "egypt": "EGY", "nigeria": "NGA", "kenya": "KEN", "south africa": "ZAF", "india": "IND",
    "pakistan": "PAK", "china": "CHN", "japan": "JPN", "south korea": "KOR", "korea": "KOR",
    "taiwan": "TWN", "hong kong": "HKG", "singapore": "SGP", "indonesia": "IDN", "vietnam": "VNM",
    "thailand": "THA", "philippines": "PHL", "malaysia": "MYS", "australia": "AUS", "new zealand": "NZL",
}


def _theme_charts(brief, by_idx, api_key, model):
    """Ask the AI for up to TABLES_PER_THEME checked tables of figures per theme. Returns
    [{"theme", "form", "title", "measure", "unit", "ordered", "parts", "points": [{"label", "series",
    "who", "method", "statistic", "n", "lo", "hi", "year", "idx", "sources", "source", "claim"}]}]."""
    by_theme = {}
    for f in brief.get("findings", []):
        if re.search(r"\d", f["claim"]):
            by_theme.setdefault(f.get("theme"), []).append(f)

    charts = []
    for t in brief.get("themes", []):
        found = {f["id"]: f for f in by_theme.get(t["title"], [])[:MAX_NUMBERED]}
        if len(found) < 2:
            print(f"   {t['title']}: fewer than 2 findings with numbers, no chart")
            continue
        reply = call_ai(
            f"RESEARCH QUESTION: {brief.get('question', '')}\n"
            f"THEME: {t['title']}\nKEY FINDING OF THIS THEME: {t.get('key_finding', '')}\n\n"
            f"Below are this theme's findings that contain numbers (ID: claim). Extract up to "
            f"{TABLES_PER_THEME} TABLES of figures that show what the evidence on this theme "
            "says, as a careful researcher would. Make a table only where at least 2 figures "
            "measure the same thing; most themes need one or two.\n\n"
            "Rules:\n"
            "- One MEASURE per table: a single quantity in one unit. Every figure in the table "
            "measures it. People's own estimates (a survey asking whether they feel faster) "
            "and measured outcomes (timed tasks, output data) are different measures, except in "
            "a table that sets the two side by side for the same group of one study (put "
            "'perceived' and 'measured' in the series column).\n"
            "- Include every figure the findings give for the measure, also those that disagree "
            "with the key finding. Do not choose figures to fit a story.\n"
            "- Write only what each finding states. If it does not say the method, the sample "
            "size or the kind of number, write 'not stated'. Do not guess.\n\n"
            "Format, exactly, for each table:\n"
            "TABLE\n"
            "MEASURE: <the quantity, with its unit in brackets>\n"
            "TITLE: <what the figures show, in plain words, at most 16 words: the range, the "
            "difference, the order or the change, and how far sources agree. Only numbers from "
            "the table. No cause unless an experiment shows it. Say what the figures show, never "
            "what the chart is, how it was made or how to read it.>\n"
            "ORDERED: <yes if the categories have a natural order, e.g. age bands or company "
            "sizes; otherwise no>\n"
            "PARTS: <yes if the figures are the parts of one whole from one source that add up "
            "to about 100%; otherwise no>\n"
            f"Then 2 to {MAX_POINTS} rows:\n"
            "- C12 | <category: who or what the figure is about (a group, a country, an answer "
            "option), at most 6 words> | <series: a second split when one category has several "
            "figures, e.g. a year, 'perceived'/'measured', 'before'/'after'; otherwise empty> | "
            "<who measured it, at most 4 words> | <method: " + " / ".join(METHODS) + "> | <kind of "
            "number: " + " / ".join(STATISTICS) + "> | <the number as written in the finding, "
            "e.g. 14 or 35-45; negative for a decrease> | <sample size, or empty> | <year, or "
            "empty>\n\n"
            "If no table fits, answer only: NONE\n\n"
            "FINDINGS:\n" + "\n".join(f"{i}: {f['claim']}" for i, f in found.items()),
            api_key, model, temp=0) or ""
        blocks = re.split(r"(?m)^\W*TABLE\W*$", reply)[1:]
        made = [c for c in (_read_table(b, found, by_idx) for b in blocks) if c][:TABLES_PER_THEME]
        if not made:
            print(f"   {t['title']}: no chart (no 2 figures measure the same thing)")
        for c in made:
            c["theme"] = t["title"]
            c["form"] = _form(c)
            charts.append(c)
    return charts


def _read_table(reply, found, by_idx):
    """Read one table from the AI's reply; drop figures and a title whose numbers don't check out."""
    def field(name):
        m = re.search(rf"^\W*{name}\W*:\s*(.+)$", reply, re.I | re.M)
        return m.group(1).strip().strip("*").strip() if m else ""

    def pick(text, allowed):
        text = text.lower()
        return next((a for a in allowed if a in text), "not stated")

    measure = field("MEASURE")
    unit = re.search(r"\(([^)]*)\)\s*$", measure)
    unit = unit.group(1) if unit else ""
    points = []
    for line in reply.splitlines():
        parts = [x.strip() for x in line.strip().lstrip("-• ").split("|")]
        if len(parts) < 7 or parts[0] not in found:
            continue
        parts += [""] * (9 - len(parts))
        f = found[parts[0]]
        m = re.fullmatch(r"([-+−]?)\s*(\d[\d.,]*)\s*%?(?:\s*(?:-|–|to)\s*(\d[\d.,]*)\s*%?)?", parts[6])
        if not m:
            continue
        sources = [by_idx[s] for s in f["sources"][:3] if s in by_idx]
        raws = [r for r in (m.group(2), m.group(3)) if r]
        # Each number must be in the finding and in one of its sources
        if not all(_in_text(r, f["claim"]) and any(_in_text(r, s.get("content", "")) for s in sources)
                   for r in raws):
            print(f"   left out of the chart (number not found in its source): {parts[0]} {parts[6]}")
            continue
        sign = -1 if m.group(1) in ("-", "−") else 1
        values = sorted(sign * float(_readings_one(r)) for r in raws)
        series, year = parts[2][:30], parts[8]
        if not year and re.fullmatch(r"(19|20)\d\d", series):
            series, year = "", series      # A year written in the series column
        year = year if re.fullmatch(r"(19|20)\d\d", year) and year in f["claim"] else ""
        n = re.sub(r"[^\d,.]", "", parts[7])
        n = n if n and _in_text(n, f["claim"]) else ""
        label = parts[1][:50]
        if any(not _in_text(x, f["claim"]) for x in re.findall(r"\d+(?:[.,]\d+)?", label)):
            label = ""
        lead = sources[0] if sources else {}
        points.append({
            "label": label or parts[0], "series": series, "claim": f["claim"],
            "who": "" if parts[3].lower().strip(".") in ("", "not stated", "unknown", "n/a") else parts[3][:30],
            "method": pick(parts[4], METHODS), "statistic": pick(parts[5], STATISTICS), "n": n,
            "lo": values[0], "hi": values[-1], "year": year,
            "idx": lead.get("idx"), "sources": set(f["sources"]),
            "source": f"{lead.get('title', '')[:80]} ({lead.get('domain', '')})",
        })
    points = points[:MAX_POINTS]
    if len(points) < 2:
        return None

    # The title may only use numbers on the chart or in its findings
    title = field("TITLE")
    allowed = set().union(*(verify.numbers(p["claim"]) for p in points))
    for p in points:
        for v in (p["lo"], p["hi"]):
            allowed |= verify._readings(f"{abs(v):g}") | {f"{round(abs(v)):g}"}   # 54.9 may be written 55
        allowed.add(p["year"])
    unknown = {x for x in re.findall(r"\d+(?:[.,]\d+)?", title)
               if not (verify._readings(x.replace(",", ".")) & allowed)}
    if unknown or not title:
        if title:
            print(f"   chart title replaced (numbers not in its figures: {', '.join(sorted(unknown))}): {title}")
        title = ""
    return {"title": title, "measure": re.sub(r"\s*\([^)]*\)\s*$", "", measure), "unit": unit,
            "ordered": field("ORDERED").lower().startswith("y"),
            "parts": field("PARTS").lower().startswith("y"), "points": points}


def _stem(word):
    """A rough word stem, to match word forms: "coding", "code" → "cod"; "developers" → "develop"."""
    for end in ("ments", "ment", "ings", "ing", "ers", "er", "ed", "es", "s", "e"):
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[:-len(end)]
    return word


def _country(label):
    """The ISO code of the one country a label names ("Developers in Japan" → "JPN"), or None."""
    hits = {code for name, code in COUNTRIES.items()
            if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", label.lower())}
    return hits.pop() if len(hits) == 1 else None


def _form(c):
    """The chart for a table, from the shape of its data (data-to-viz.com):
      line        the same category in 2+ years
      dumbbell    two figures per category from one study (perceived vs measured, before vs after)
      heatmap     3+ categories × 3+ series from one study
      grouped     2 series, or a small grid, from one study
      stacked     parts of one whole (up to 6)
      treemap     parts of one whole (7 or more)
      map         one figure per country
      bar         one figure per category from one study (up to 4 categories)
      lollipop    one figure per category from one study (5 or more)
      box         one measure from many sources (DISTRIBUTION_MIN or more figures)
      dots        one measure from a few sources (a Cleveland dot plot)
    Comparisons need figures from one study; otherwise the figures are shown as dots or a box.
    """
    pts = c["points"]
    one_study = bool(set.intersection(*(p["sources"] for p in pts)))
    by_label = {}
    for p in pts:
        by_label.setdefault(p["label"], []).append(p)
    years = {p["year"] for p in pts if p["year"]}
    if len(years) >= 2 and any(len({p["year"] for p in v if p["year"]}) >= 2 for v in by_label.values()):
        return "line"
    series = {p["series"] for p in pts if p["series"]}
    if len(series) >= 2:
        pairs_one_study = all(set.intersection(*(p["sources"] for p in v)) for v in by_label.values())
        if len(series) == 2 and pairs_one_study:
            return "dumbbell"
        if one_study:
            return "heatmap" if len(series) >= 3 and len(by_label) >= 3 else "grouped"
        for p in pts:   # Different studies: each figure on its own row
            p["label"] = f"{p['label']}, {p['series']}"
        return "box" if len(pts) >= DISTRIBUTION_MIN else "dots"
    if c["parts"] and one_study and 90 <= sum(p["hi"] for p in pts) <= 110:
        return "stacked" if len(pts) <= 6 else "treemap"
    if len(pts) >= 3 and all(_country(p["label"]) for p in pts) and len({_country(p["label"]) for p in pts}) == len(pts):
        return "map"
    if one_study:
        return "bar" if len(pts) <= 4 else "lollipop"
    return "box" if len(pts) >= DISTRIBUTION_MIN else "dots"


def _draw(c, go, colors, this_year):
    """Draw one findings chart in the form _form chose; return (figure, slide needs a legend)."""
    ACCENT, DARK, MUTED, GRAY, BLUE = (colors[k] for k in ("accent", "dark", "muted", "gray", "blue"))
    pts, unit, form = c["points"], c["unit"], c["form"]
    signed = any(p["lo"] < 0 for p in pts)
    share = bool(re.search(r"%\s*of", unit)) and max(p["hi"] for p in pts) <= 100 and not signed
    axis = f"{c['measure']} ({unit})" if unit else c["measure"]

    def show(p):
        """A figure as text: "up to 55%", "35-45%", "+6% to +9%"."""
        if p["lo"] == p["hi"]:
            text = _fmt(p["lo"], unit, signed)
        elif signed:
            text = f"{_fmt(p['lo'], unit, signed)} to {_fmt(p['hi'], unit, signed)}"
        else:
            text = f"{p['lo']:g}-{_fmt(p['hi'], unit)}"
        return ("up to " if p["statistic"] == "up to" else "") + text

    for p in pts:
        extra = ", ".join(x for x in (p["who"], f"n={p['n']}" if p["n"] else "") if x)
        p["row"] = _wrap(p["label"] + (f" ({extra})" if extra and form in ("dots", "box") else ""))
        p["hover"] = (f"{p['label']}{' (' + p['series'] + ')' if p['series'] else ''}: {show(p)}"
                      f"<br>Measured by: {p['who'] or 'not stated'}"
                      f"<br>Method: {p['method']}; number: {p['statistic']}"
                      + (f"; sample: {p['n']}" if p["n"] else "")
                      + (f"; year: {p['year']}" if p["year"] else "") + f"<br>{p['source']}")
    fig = go.Figure()
    legend = False
    # Marker per method: filled for measured outcomes, open for claims, forecasts and not stated
    style = {"experiment": ("diamond", DARK), "observational data": ("circle", DARK),
             "survey": ("square", BLUE), "vendor claim": ("triangle-up-open", ACCENT),
             "forecast": ("x-thin-open", GRAY), "not stated": ("circle-open", GRAY)}

    def value_axis(horizontal=True):
        """Value axis: 0-100 for shares, otherwise from 0 (or the lowest value) with room for labels."""
        lo, hi = min(0, min(p["lo"] for p in pts)), max(p["hi"] for p in pts)
        span = (hi - lo) or 1
        rng = [0, 115] if share else [lo - (0.3 if lo < 0 else 0.05) * span, hi + 0.3 * span]
        (fig.update_xaxes if horizontal else fig.update_yaxes)(range=rng, title=axis)
        if signed:
            (fig.add_vline if horizontal else fig.add_hline)(0, line_color=GRAY, line_width=1)

    if form in ("dots", "box"):
        # Dots: one row per figure, by method then value. Box: spread of all figures as points
        pts.sort(key=lambda p: (METHODS.index(p["method"]), p["hi"]))
        if form == "box":
            fig.add_trace(go.Box(x=[(p["lo"] + p["hi"]) / 2 for p in pts], y0=0, orientation="h",
                                 boxpoints=False, fillcolor="rgba(0,0,0,0)", line_color=GRAY,
                                 name="Spread", showlegend=False, hoverinfo="skip"))
            # Jitter so equal values stay visible
            for i, p in enumerate(pts):
                p["jitter"] = ((i * 7) % 11 - 5) / 14
            fig.update_yaxes(visible=False, range=[-0.8, 0.8])
            rows = None
        else:
            rows = _unique([p["row"] for p in pts])
            for p, row in zip(pts, rows):
                p["row"] = row
        for method in METHODS:
            mp = [p for p in pts if p["method"] == method]
            if not mp:
                continue
            symbol, color = style[method]
            ys = [p["jitter"] for p in mp] if form == "box" else [p["row"] for p in mp]
            for p, y in zip(mp, ys):
                if p["lo"] != p["hi"] and form == "dots":
                    fig.add_trace(go.Scatter(x=[p["lo"], p["hi"]], y=[y, y], mode="lines",
                                             line=dict(color=color, width=3), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[p["hi"] if form == "dots" else (p["lo"] + p["hi"]) / 2 for p in mp], y=ys,
                name=method.capitalize(), mode="markers" + ("+text" if form == "dots" else ""),
                marker=dict(symbol=symbol, size=13 if form == "dots" else 11, color=color,
                            line=dict(width=2, color=color), opacity=1 if form == "dots" else 0.8),
                text=[show(p) for p in mp] if form == "dots" else None, textposition="middle right",
                cliponaxis=False, hovertext=[p["hover"] for p in mp], hoverinfo="text"))
        # Median line from 3+ figures, leaving out "up to" maxima
        typical = sorted((p["lo"] + p["hi"]) / 2 for p in pts if p["statistic"] != "up to")
        if len(typical) >= 3:
            k = len(typical) // 2
            median = typical[k] if len(typical) % 2 else (typical[k - 1] + typical[k]) / 2
            fig.add_vline(x=median, line_dash="dash", line_color=GRAY, line_width=1,
                          annotation_text=f"median {_fmt(round(median, 1), unit, signed)}",
                          annotation_position="top", annotation_font_color=GRAY)
        legend = True
        if rows:
            fig.update_yaxes(categoryorder="array", categoryarray=rows, autorange="reversed")
        value_axis()

    elif form in ("bar", "lollipop"):
        # Largest first unless the categories have their own order
        if not c["ordered"]:
            pts.sort(key=lambda p: -p["hi"])
        rows = _unique([p["row"] for p in pts])
        if form == "bar":
            fig.add_trace(go.Bar(
                y=rows, x=[p["hi"] - p["lo"] if p["lo"] != p["hi"] else p["hi"] for p in pts],
                base=[p["lo"] if p["lo"] != p["hi"] else 0 for p in pts], orientation="h",
                marker_color=DARK, text=[show(p) for p in pts], textposition="outside",
                cliponaxis=False, hovertext=[p["hover"] for p in pts], hoverinfo="text", showlegend=False))
            fig.update_layout(bargap=0.35)
        else:
            for p, row in zip(pts, rows):
                fig.add_trace(go.Scatter(x=[p["lo"] if p["lo"] != p["hi"] else 0, p["hi"]], y=[row, row],
                                         mode="lines", line=dict(color=GRAY, width=2), showlegend=False,
                                         hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[p["hi"] for p in pts], y=rows, mode="markers+text",
                                     marker=dict(size=12, color=DARK), text=[show(p) for p in pts],
                                     textposition="middle right", cliponaxis=False,
                                     hovertext=[p["hover"] for p in pts], hoverinfo="text", showlegend=False))
        fig.update_yaxes(categoryorder="array", categoryarray=rows, autorange="reversed")
        value_axis()

    elif form == "grouped":
        # One group per category, one color per series
        series = list(dict.fromkeys(p["series"] for p in pts))
        palette = dict(zip(series, [DARK, BLUE, MUTED, GRAY, ACCENT]))
        for sname in series:
            sp = [p for p in pts if p["series"] == sname]
            fig.add_trace(go.Bar(
                y=[p["row"] for p in sp], x=[p["hi"] for p in sp], orientation="h", name=sname or "value",
                marker_color=palette.get(sname, GRAY), text=[show(p) for p in sp], textposition="outside",
                cliponaxis=False, hovertext=[p["hover"] for p in sp], hoverinfo="text"))
        legend = True
        fig.update_layout(barmode="group", yaxis_autorange="reversed", bargap=0.3)
        value_axis()

    elif form == "dumbbell":
        # One row per category, a line between its two series
        rows = list(dict.fromkeys(p["label"] for p in pts))
        sides = list(dict.fromkeys(p["series"] for p in pts))[:2]
        for r in rows:
            pair = [p for p in pts if p["label"] == r]
            fig.add_trace(go.Scatter(x=[p["hi"] for p in pair], y=[_wrap(r)] * len(pair), mode="lines",
                                     line=dict(color=GRAY, width=3), showlegend=False, hoverinfo="skip"))
        for side, color in zip(sides, [MUTED, DARK]):
            sp = [p for p in pts if p["series"] == side]
            fig.add_trace(go.Scatter(
                x=[p["hi"] for p in sp], y=[_wrap(p["label"]) for p in sp], name=side, mode="markers+text",
                marker=dict(size=14, color=color), text=[show(p) for p in sp], textposition="top center",
                cliponaxis=False, hovertext=[p["hover"] for p in sp], hoverinfo="text"))
        legend = True
        fig.update_yaxes(autorange="reversed")
        value_axis()

    elif form == "heatmap":
        rows = list(dict.fromkeys(p["label"] for p in pts))
        cols = list(dict.fromkeys(p["series"] for p in pts))
        cell = {(p["label"], p["series"]): p for p in pts}
        z = [[cell[(r, k)]["hi"] if (r, k) in cell else None for k in cols] for r in rows]
        fig.add_trace(go.Heatmap(
            z=z, x=cols, y=[_wrap(r) for r in rows], colorscale=[[0, "#F1F6F9"], [1, DARK]],
            text=[[show(cell[(r, k)]) if (r, k) in cell else "" for k in cols] for r in rows],
            texttemplate="%{text}", hovertext=[[cell[(r, k)]["hover"] if (r, k) in cell else "" for k in cols]
                                               for r in rows], hoverinfo="text", colorbar=dict(title=unit)))
        fig.update_yaxes(autorange="reversed")

    elif form == "stacked":
        # One 100% bar split into its parts, darkest first
        shades = ["#1D3557", "#457B9D", "#7FA7C9", "#A8DADC", "#D3E9EA", "#E8E8E8"]
        for p, color in zip(pts, shades):
            fig.add_trace(go.Bar(
                y=[c["measure"]], x=[p["hi"]], name=p["label"], orientation="h", marker_color=color,
                text=[show(p)], textposition="inside", insidetextanchor="middle",
                hovertext=[p["hover"]], hoverinfo="text"))
        legend = True
        fig.update_layout(barmode="stack", bargap=0.5, legend_traceorder="normal")
        fig.update_xaxes(range=[0, max(100, sum(p["hi"] for p in pts))], title=axis)
        fig.update_yaxes(showticklabels=False)

    elif form == "treemap":
        fig.add_trace(go.Treemap(
            labels=[f"{p['label']}<br>{show(p)}" for p in pts], parents=[""] * len(pts),
            values=[p["hi"] for p in pts], marker=dict(colors=[p["hi"] for p in pts],
                                                       colorscale=[[0, "#D3E9EA"], [1, DARK]]),
            hovertext=[p["hover"] for p in pts], hoverinfo="text", textinfo="label"))

    elif form == "map":
        # One figure per country, darker for higher values
        fig.add_trace(go.Choropleth(
            locations=[_country(p["label"]) for p in pts], z=[p["hi"] for p in pts],
            text=[p["hover"] for p in pts], hoverinfo="text",
            colorscale=[[0, "#D3E9EA"], [1, DARK]], colorbar=dict(title=unit)))
        lo, hi = min(p["hi"] for p in pts), max(p["hi"] for p in pts)
        fig.add_trace(go.Scattergeo(
            locations=[_country(p["label"]) for p in pts], text=[show(p) for p in pts], mode="text",
            textfont=dict(size=13, color=["white" if (p["hi"] - lo) / ((hi - lo) or 1) > 0.5 else DARK
                                          for p in pts]),
            hoverinfo="skip", showlegend=False))
        fig.update_geos(showframe=False, showland=True, landcolor="#F4F4F4", showcountries=True,
                        countrycolor="#B0B0B0", showcoastlines=True, coastlinecolor="#B0B0B0",
                        projection_type="natural earth", fitbounds=False, projection_rotation_lon=0,
                        lataxis_range=[-58, 85], lonaxis_range=[-180, 180])

    else:
        # Line chart: up to MAX_LINES categories with the most years; forecasts get open markers
        lines = {}
        for p in pts:
            if p["year"]:
                lines.setdefault(p["label"], []).append(p)
        lines = dict(sorted(lines.items(), key=lambda kv: -len(kv[1]))[:MAX_LINES])
        palette = iter([DARK, BLUE, ACCENT, MUTED, GRAY])
        for name, sp in lines.items():
            sp.sort(key=lambda p: p["year"])
            future = [int(p["year"]) > this_year or p["method"] == "forecast" for p in sp]
            fig.add_trace(go.Scatter(
                x=[int(p["year"]) for p in sp], y=[p["hi"] for p in sp], name=_wrap(name, 40),
                mode="lines+markers+text", line=dict(color=next(palette, GRAY), width=2),
                marker=dict(size=10, symbol=["circle-open" if f else "circle" for f in future]),
                text=[show(p) + (" (forecast)" if f else "") for p, f in zip(sp, future)],
                textposition="top center", cliponaxis=False, hovertext=[p["hover"] for p in sp],
                hoverinfo="text"))
        legend = len(lines) > 1
        fig.update_xaxes(title="", dtick=1, tickformat="d")
        if share:
            fig.update_yaxes(range=[0, 105])
        fig.update_yaxes(title=axis)
    return fig, legend


def _in_text(raw, text):
    """True if the number `raw` ("55", "1", "0,47") is written in `text`."""
    if verify._readings(raw.replace(",", ".")) & verify.numbers(text):
        return True
    # numbers() skips single digits, so look for those directly
    return bool(re.search(rf"(?<![\d.,]){re.escape(raw)}(?![\d])", text or ""))


def _readings_one(raw):
    """One number from its text: "1,000" → 1000, "0,47" → 0.47."""
    r = sorted(verify._readings(raw))
    return r[0] if len(r) == 1 else raw.replace(",", "")


def _wrap(text, width=28):
    """Break a long label into lines of about `width` characters for the chart axis."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "<br>".join(lines + [line])


def _fmt(value, unit, signed=False):
    """A figure as text: 84, "%" → "84%"; signed → "+84%"; 3.5, "hours" → "3.5 hours"."""
    number = f"{value:+g}" if signed else f"{value:g}"
    return f"{number}%" if "%" in unit else f"{number} {unit}".strip()


def _unique(labels):
    """Make chart labels unique: a second "GitHub survey" becomes "GitHub survey (2)"."""
    seen, out = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out
