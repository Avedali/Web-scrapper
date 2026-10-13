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

from core import (call_ai, source_label, LANGUAGES, apa_reference, apa_in_text,
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
    tests_path = os.path.join(args.corpus, "stats_results.json")   # Test results from stats, if run
    tests = json.load(open(tests_path, encoding="utf-8")) if os.path.exists(tests_path) else {}
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

        api_key = getattr(args, "key", None) or os.environ.get("OPENROUTER_API_KEY")
        colors = {"accent": ACCENT, "dark": DARK, "muted": MUTED, "gray": GRAY, "blue": "#457B9D"}

        def draw_all(charts_list):
            """Draw and save each chart, with its sources cited APA style."""
            for c in charts_list:
                fig, legend = _draw(c, go, colors, datetime.date.today().year)
                srcs = [by_idx[i] for i in dict.fromkeys(p["idx"] for p in c["points"]) if i in by_idx]
                title = c["title"] or c["measure"]
                if not re.search(r"\d", title) and len({p["idx"] for p in c["points"]}) > 1:
                    # A title that only names the topic: give the range the figures span
                    lo, hi = min(p["lo"] for p in c["points"]), max(p["hi"] for p in c["points"])
                    title = (f"{c['measure']}: {_fmt(lo, c['unit'])} to {_fmt(hi, c['unit'])} "
                             f"across {len({p['idx'] for p in c['points']})} sources")
                save(fig, re.sub(r"\W+", "_", f"{c['theme']} {c['form']}".lower()).strip("_")[:50],
                     f"{c['theme']}: {title}", _wrap(title, 70), keep_legend=legend,
                     note=_wrap(f"Sources: {apa_in_text(srcs)}", 150),
                     paper_note="Sources:<br>" + "<br>".join(
                         _wrap(re.sub(r"\*([^*]+)\*", r"<i>\1</i>", apa_reference(x)), 120) for x in srcs))

        # ── A chart on request ("adoption by country"): only that, in requested/ ─
        request = (getattr(args, "question", None) or "").strip()
        if request:
            if not api_key:
                sys.exit("A chart on request needs the API key (run install.py).")
            folder = re.sub(r"\W+", "_", request.lower()).strip("_")[:40] or "request"
            paper_root = os.path.join(args.corpus, "charts_paper", "requested", folder)
            slide_root = os.path.join(args.corpus, "charts_slide", "requested", folder)
            asked, matched = _request_brief(brief, request)
            print(f"   Request: {request}\n   {matched} findings with numbers match it")
            made = _theme_charts(asked, by_idx, api_key, args.model) if matched >= 2 else []
            if not made:
                print(f"\n   Not enough data for a chart on '{request}': "
                      + ("fewer than 2 findings with numbers match it." if matched < 2 else
                         f"{matched} findings match, but no 2 of them measure the same thing.")
                      + "\n   Try a broader request, or run deepen to look for more evidence.")
                return
            draw_all(made)
            print(f"\n   {charts} chart(s) saved in {slide_root}/ and {paper_root}/")
            return

        # ── Up to 3 charts per theme, form chosen by _form ──────────
        if not api_key:
            print("   No API key: skipping the findings charts (they need one AI call per theme)")
        else:
            draw_all(_theme_charts(brief, by_idx, api_key, args.model))

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
        test = tests.get("vendor_vs_other")
        if is_vendor.any() and (~is_vendor).any():
            v_mean = df.loc[is_vendor, "sentiment"].mean()
            o_mean = df.loc[~is_vendor, "sentiment"].mean()
            slide_title = (f"Sources from AI tool makers average {v_mean:+.2f} sentiment.<br>"
                           f"Other sources average {o_mean:+.2f}"
                           + (f" ({_verdict(test['p'])})" if test else ""))
        else:
            slide_title = "Sentiment by source type"

        save(fig, "sentiment_by_source_type",
             f"Sentiment of each source, by type of source (N={n_total})"
             + (f". Tool makers against all others: Mann-Whitney U test, {_p_text(test['p'])}" if test else ""),
             slide_title, appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 2: Sentiment by language (lollipop chart)
    # ════════════════════════════════════════════════════════════════

    if "sentiment" in df.columns and "lang" in df.columns:

        data = df.groupby("lang").agg(
            m=("sentiment", "mean"),
            n=("sentiment", "count"),
        ).reset_index()

        test = tests.get("languages")
        # The languages the stats test compared (3+ sources each), else those with 2+
        keep = data["lang"].isin(test["languages"]) if test else data["n"] >= 2
        data = data[keep].sort_values("m")
        data["lang"] = [LANGUAGES.get(c, {}).get("name", c) for c in data["lang"]]   # "zh" → "Chinese"

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
                    textposition="middle left" if row["m"] < 0 else "middle right",
                    showlegend=False,
                ))

            fig.add_vline(x=0, line_dash="solid", line_color=DARK, line_width=1)

            lo_x, hi_x = min(0, data["m"].min()), max(0, data["m"].max())
            span = (hi_x - lo_x) or 1
            fig.update_xaxes(range=[lo_x - (0.3 if lo_x < 0 else 0.05) * span, hi_x + 0.3 * span])

            lo, hi = data.iloc[0], data.iloc[-1]
            save(fig, "sentiment_by_language",
                 "Average sentiment of the sources in each language"
                 + (f". Kruskal-Wallis test across languages: {_p_text(test['p'])}" if test else ""),
                 f"Most positive: {hi['lang']} ({hi['m']:+.2f}). "
                 f"Least positive: {lo['lang']} ({lo['m']:+.2f})"
                 + (f"<br>Languages differ ({_verdict(test['p'])})" if test and test["p"] < 0.05 else
                    f"<br>No clear difference between languages ({_p_text(test['p'])})" if test else ""),
                 appendix=True)

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
MAX_POINTS = 30       # Figures in one table
TABLES_PER_THEME = 3  # Tables of figures the AI may extract for one theme
DISTRIBUTION_MIN = 8  # Figures of one category that make a box plot
GROUP_MIN = 5         # Figures per category for a box plot per category
VIOLIN_MIN = 12       # Figures per category for a violin plot
HISTOGRAM_MIN = 15    # Figures of one category for a histogram
BAR_MAX = 8           # Categories a bar chart shows; more make a lollipop chart
MAX_LINES = 5         # Lines in one line chart

# What a table's figures are, as the AI says it (the chart is then chosen from the data)
SHAPES = {
    "compare": "figures for different groups, options or sources, to compare",
    "parts": "the parts of one whole that add up to about 100% (or several wholes, one per series)",
    "nested parts": "parts of a whole where each part belongs to a larger group, given in the series column",
    "stages": "successive stages where each is a subset of the one before (heard of → tried → use daily)",
    "change": "a starting total, the changes that add to or take from it, and the end total "
              "(write 'total' in the series column of the start and end rows)",
    "flow": "amounts moving from one group to another: the category is where from, the series where to",
    "relation": "two or three different measures for the same items, to see how they relate "
                "(the second in the x column, a third in the size column)",
}

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
        if sum(len(re.findall(r"\d+(?:[.,]\d+)?", f["claim"])) for f in found.values()) < 2:
            print(f"   {t['title']}: fewer than 2 figures, no chart")
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
            "SHAPE: <what the figures are, one of: " + "; ".join(f"{k} = {v}" for k, v in SHAPES.items())
            + ". Most tables are compare.>\n"
            "MEASURE X: <for relation only: the second measure, with its unit in brackets>\n"
            "MEASURE SIZE: <for relation only, if there is a third measure: it, with its unit in brackets>\n"
            f"Then 2 to {MAX_POINTS} rows:\n"
            "- C12 | <category: who or what the figure is about (a group, a country, an answer "
            "option), at most 6 words> | <series: a second split when one category has several "
            "figures, e.g. a year, 'perceived'/'measured', 'before'/'after'; otherwise empty> | "
            "<who measured it, at most 4 words> | <method: " + " / ".join(METHODS) + "> | <kind of "
            "number: " + " / ".join(STATISTICS) + "> | <the number as written in the finding, "
            "e.g. 14 or 35-45; negative for a decrease> | <sample size, or empty> | <year, or "
            "empty> | <x: for relation, the second measure's number; otherwise empty> | <size: for "
            "relation, the third measure's number; otherwise empty>\n\n"
            "If no table fits, answer only: NONE\n\n"
            "FINDINGS:\n" + "\n".join(f"{i}: {f['claim']}" for i, f in found.items()),
            api_key, model, temp=0) or ""
        blocks = re.split(r"(?m)^\W*TABLE\W*$", reply)[1:]
        made = [c for c in (_read_table(b, found, by_idx) for b in blocks) if c][:TABLES_PER_THEME]
        made = [c for c in (_same_measure(c, api_key, model) for c in made) if c]
        if not made:
            print(f"   {t['title']}: no chart (no 2 figures measure the same thing)")
        for c in made:
            c["theme"] = t["title"]
            c["form"] = _form(c)
            charts.append(c)
    return charts


REQUEST_MATCH = 0.35   # Meaning-similarity a finding needs to count as about the request


def _request_brief(brief, request):
    """
    A one-theme brief for a chart on request: the brief's findings with numbers that are
    about the request (by meaning when the matching model is installed, else by shared
    words), plus every finding of a theme whose title the request names.
    Returns (brief, number of matching findings).
    """
    numbered = [f for f in brief.get("findings", []) if re.search(r"\d", f["claim"])]
    named = {t["title"] for t in brief.get("themes", [])
             if request.lower() in t["title"].lower() or t["title"].lower() in request.lower()}
    model = verify._match_model()
    if model is not None and numbered:
        from sentence_transformers import util
        score = util.cos_sim(model.encode([request], convert_to_tensor=True),
                             model.encode([f["claim"] for f in numbered], convert_to_tensor=True))[0].tolist()
        need = REQUEST_MATCH
    else:
        words = set(re.findall(r"\w{4,}", request.lower()))
        score = [len(words & set(re.findall(r"\w{4,}", f["claim"].lower()))) for f in numbered]
        need = max(1, len(words) // 2)
    # Most relevant first, so the cap on findings shown to the AI drops the least relevant
    ranked = sorted(((sc + (1 if f.get("theme") in named else 0), f) for sc, f in zip(score, numbered)
                     if sc >= need or f.get("theme") in named), key=lambda x: -x[0])
    found = [dict(f, theme=request) for _, f in ranked[:MAX_NUMBERED]]
    return ({"question": brief.get("question", ""),
             "themes": [{"title": request, "key_finding": f"What the evidence says about {request}"}],
             "findings": found}, len(found))


def _read_table(reply, found, by_idx):
    """Read one table from the AI's reply; drop figures and a title whose numbers don't check out."""
    def field(name):
        m = re.search(rf"^\W*{name}\W*:\s*(.+)$", reply, re.I | re.M)
        return m.group(1).strip().strip("*").strip() if m else ""

    def pick(text, allowed):
        text = text.lower()
        return next((a for a in allowed if a in text), "not stated")

    def measure_unit(text):
        """("Share of developers", "% of developers") from "Share of developers (percentage of developers)"."""
        unit = re.search(r"\(([^)]*)\)\s*$", text)
        unit = unit.group(1) if unit else ""
        unit = re.sub(r"^\s*(percentage|percent|per cent|pct)(?=\s|$)", "%", unit, flags=re.I)
        return re.sub(r"\s*\([^)]*\)\s*$", "", text), unit

    def checked(raw, f, sources):
        """A number from a cell, if it is written in the finding and in one of its sources."""
        m = re.fullmatch(r"([-+−]?)\s*(\d[\d.,]*)\s*%?", raw.strip())
        if not m or not (_in_text(m.group(2), f["claim"]) and any(_in_text(m.group(2), s.get("content", "")) for s in sources)):
            return None
        return (-1 if m.group(1) in ("-", "−") else 1) * float(_readings_one(m.group(2)))

    measure, unit = measure_unit(field("MEASURE"))
    shape = pick(field("SHAPE"), sorted(SHAPES, key=len, reverse=True))
    shape = "compare" if shape == "not stated" else shape
    if field("PARTS").lower().startswith("y") and shape == "compare":
        shape = "parts"
    points = []
    for line in reply.splitlines():
        parts = [x.strip() for x in line.strip().lstrip("-• ").split("|")]
        if len(parts) < 7 or parts[0] not in found:
            continue
        parts += [""] * (11 - len(parts))
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
            "cite": apa_in_text([lead]) if lead else "",
            "x": checked(parts[9], f, sources) if parts[9] else None,
            "size": checked(parts[10], f, sources) if parts[10] else None,
        })
    points = points[:MAX_POINTS]
    if len(points) < 2:
        return None

    measure_x, unit_x = measure_unit(field("MEASURE X"))
    measure_size, unit_size = measure_unit(field("MEASURE SIZE"))
    return {"title": _checked_title(field("TITLE"), points), "measure": measure, "unit": unit,
            "ordered": field("ORDERED").lower().startswith("y"), "shape": shape,
            "parts": shape in ("parts", "nested parts"), "points": points,
            "measure_x": measure_x, "unit_x": unit_x, "measure_size": measure_size, "unit_size": unit_size}


def _checked_title(title, points):
    """The title, or "" if it uses a number that is not on the chart or in its findings."""
    allowed = set().union(*(verify.numbers(p["claim"]) for p in points))
    for p in points:
        for v in (p["lo"], p["hi"], p.get("x"), p.get("size")):
            if v is not None:
                allowed |= verify._readings(f"{abs(v):g}") | {f"{round(abs(v)):g}"}   # 54.9 may be written 55
        allowed.add(p["year"])
    unknown = {x for x in re.findall(r"\d+(?:[.,]\d+)?", title)
               if not (verify._readings(x.replace(",", ".")) & allowed)}
    if unknown and title:
        print(f"   chart title replaced (numbers not in its figures: {', '.join(sorted(unknown))}): {title}")
    return "" if unknown else title


def _same_measure(c, api_key, model):
    """Drop figures that measure something else than the table's MEASURE (e.g. a usage share
    put in a table of perceived benefits). Returns the table, or None if fewer than 2 are left."""
    pts = c["points"]
    reply = call_ai(
        f"MEASURE: {c['measure']}" + (f" ({c['unit']})" if c["unit"] else "") + "\n\n"
        "Each line below is a figure filed under this measure, with the finding it comes from. "
        "Say for each whether the figure measures exactly this quantity: the same thing counted, "
        "among the same kind of people or units. A share of people who use something does not "
        "measure what they gain from it; a share who feel an effect is not a measured effect.\n\n"
        "Answer one line per figure, exactly: <number> | yes or no\n\n"
        + "\n".join(f"{i}. {p['label']}: {p['lo']:g}" + ("" if p["lo"] == p["hi"] else f" to {p['hi']:g}")
                    + f" -- finding: {p['claim']}" for i, p in enumerate(pts, 1)),
        api_key, model, temp=0) or ""
    off = {int(m.group(1)) for m in re.finditer(r"(?m)^\W*(\d+)\W*\|\s*no\b", reply, re.I)}
    for i in sorted(off):
        if i <= len(pts):
            print(f"   left out of the chart (measures something else): {pts[i - 1]['label']}")
    c["points"] = [p for i, p in enumerate(pts, 1) if i not in off]
    if len(c["points"]) < 2:
        return None
    c["title"] = _checked_title(c["title"], c["points"])
    return c


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
    """The chart for a table, from the shape of its data, by the data-to-viz.com decision tree
    (the R Graph Gallery's guide) and plotly's chart types:
      relation    two measures per item: scatter; a third: bubble; one item over 3+ years: connected scatter
      flow        from → to amounts: sankey
      stages      shrinking stages of one study: funnel
      change      start, changes, end of one study: waterfall
      time        3+ years: line; parts of a whole over 3+ years: stacked area; 2 years: slope chart
      parts       one whole: donut (2-4 parts), stacked bar (5-6), treemap (7+); several wholes:
                  100% stacked bars; parts in larger groups: sunburst
      subgroups   two series paired in one study: dumbbell; up to 3 items × 5+ series: radar;
                  3+ × 3+: heatmap; otherwise grouped bars
      map         one value per country: choropleth for shares and rates, bubble map for counts
      spread      one category with HISTOGRAM_MIN+ figures: histogram; DISTRIBUTION_MIN+: box;
                  several categories with GROUP_MIN+ figures each: box per category (VIOLIN_MIN+: violin)
      compare     one value per category: bar (up to BAR_MAX) or lollipop
    One category from several sources: the sources are the categories ("JetBrains, 2025").
    Several categories with several sources each: one bar per category and source, side by side.
    """
    pts, shape = c["points"], c.get("shape", "compare")
    # One wording per category: "Increased" and "Increasing productivity" are the same answer
    first = {}
    for p in pts:
        p["label"] = first.setdefault(tuple(sorted(_stem(w) for w in re.findall(r"\w+", p["label"].lower()))), p["label"])
    by_label = {}
    for p in pts:
        by_label.setdefault(p["label"], []).append(p)
    one_study = bool(set.intersection(*(p["sources"] for p in pts)))
    years_of = {k: {p["year"] for p in v if p["year"]} for k, v in by_label.items()}
    series = {p["series"] for p in pts if p["series"]}

    if shape == "relation" and sum(p["x"] is not None for p in pts) >= 3:
        c["points"] = pts = [p for p in pts if p["x"] is not None]
        if any(len({p["year"] for p in v if p["x"] is not None and p["year"]}) >= 3 for v in by_label.values()):
            return "connected"
        return "bubble" if all(p["size"] is not None and p["size"] > 0 for p in pts) else "scatter"
    if shape == "flow" and all(p["series"] for p in pts):
        return "sankey"
    if shape == "stages" and one_study and len(pts) >= 3:
        if all(a["hi"] >= b["hi"] for a, b in zip(pts, pts[1:])):
            return "funnel"
    if shape == "change" and one_study and sum(p["series"].lower() != "total" for p in pts) >= 2:
        return "waterfall"

    # Time
    if c["parts"] and len({p["year"] for p in pts if p["year"]}) >= 3 and len(by_label) >= 2:
        return "area"
    if any(len(y) >= 3 for y in years_of.values()):
        return "line"
    if any(len(y) == 2 for y in years_of.values()):
        return "slope" if sum(len(y) == 2 for y in years_of.values()) >= 2 else "line"

    # Parts of a whole
    if c["parts"]:
        if shape == "nested parts" and series:
            return "sunburst"
        wholes = {}
        for p in pts:
            wholes[p["series"]] = wholes.get(p["series"], 0) + p["hi"]
        if len(wholes) >= 2 and all(80 <= v <= 120 for v in wholes.values()):
            return "stacked100"
        if one_study and 90 <= sum(p["hi"] for p in pts) <= 110:
            return "donut" if len(pts) <= 4 else "stacked" if len(pts) <= 6 else "treemap"

    # Subgroups: categories × series
    if len(series) >= 2:
        paired = all(set.intersection(*(p["sources"] for p in v)) for v in by_label.values())
        if len(series) == 2 and paired:
            return "dumbbell"
        full = all(len({p["series"] for p in v}) == len(series) for v in by_label.values())
        if len(by_label) <= 3 and len(series) >= 5 and full:
            return "radar"
        return "heatmap" if len(series) >= 3 and len(by_label) >= 3 else "grouped"

    # Geography
    if len(pts) >= 3 and all(_country(p["label"]) for p in pts) and len({_country(p["label"]) for p in pts}) == len(pts):
        rate = "%" in c["unit"] or re.search(r"\bper\b|rate|average|share|index|score", c["unit"] + " " + c["measure"], re.I)
        return "map" if rate else "bubblemap"

    # Spread of many figures
    if len(by_label) == 1 and len(pts) >= HISTOGRAM_MIN:
        return "histogram"
    if len(by_label) == 1 and len(pts) >= DISTRIBUTION_MIN:
        return "box"
    if len(by_label) >= 2 and all(len(v) >= GROUP_MIN for v in by_label.values()):
        return "violin" if all(len(v) >= VIOLIN_MIN for v in by_label.values()) else "boxes"

    # Compare one value per category
    if len(by_label) == 1:
        for p in pts:   # The sources are the categories
            p["label"], p["by_source"] = _source_label(p), True
    elif len(by_label) < len(pts):
        order = list(by_label)
        for p in pts:
            p["label"] = f"{p['label']}: {_source_label(p)}"
            p["by_source"] = True
        pts.sort(key=lambda p: (order.index(p["label"].rsplit(": ", 1)[0]), -p["hi"]))
        c["ordered"] = True   # Keep each category's bars together
    return "bar" if len(pts) <= BAR_MAX else "lollipop"


def _source_label(p):
    """A figure's source as a row label: who measured it, else the cited source ("JetBrains, 2025")."""
    year = p["year"] or (re.findall(r"\d{4}|n\.d\.", p.get("cite", "")) or [""])[-1]
    return ", ".join(x for x in (p["who"], year) if x) if p["who"] else p.get("cite", "").strip("()") or "source"


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
        return ("up to " if p["statistic"] == "up to" and p["lo"] == p["hi"] else "") + text

    several = len({p["idx"] for p in pts}) > 1
    for p in pts:
        p["row"] = _wrap(p["label"])
        # Figures from different sources: each row says which source it comes from
        if several and form in ("bar", "lollipop") and p.get("cite") and not p.get("by_source"):
            p["row"] += f"<br><span style='font-size:11px;color:#6B6B6B'>{_wrap(p['cite'], 34)}</span>"
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

    if form == "box":
        # Box of all figures, every figure shown as a point (jittered so equal values stay visible)
        fig.add_trace(go.Box(x=[(p["lo"] + p["hi"]) / 2 for p in pts], y0=0, orientation="h",
                             boxpoints=False, fillcolor="rgba(0,0,0,0)", line_color=GRAY,
                             name="Spread", showlegend=False, hoverinfo="skip"))
        for i, p in enumerate(pts):
            p["jitter"] = ((i * 7) % 11 - 5) / 14
        for method in METHODS:
            mp = [p for p in pts if p["method"] == method]
            if mp:
                symbol, color = style[method]
                fig.add_trace(go.Scatter(
                    x=[(p["lo"] + p["hi"]) / 2 for p in mp], y=[p["jitter"] for p in mp],
                    name=method.capitalize(), mode="markers",
                    marker=dict(symbol=symbol, size=11, color=color, line=dict(width=2, color=color), opacity=0.8),
                    hovertext=[p["hover"] for p in mp], hoverinfo="text"))
        fig.update_yaxes(visible=False, range=[-0.8, 0.8])
        legend = True
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
            # Stem from 0; a range ("20-40%") is a thick segment on the stem
            for p, row in zip(pts, rows):
                fig.add_trace(go.Scatter(x=[0, p["hi"]], y=[row, row], mode="lines",
                                         line=dict(color=GRAY, width=2), showlegend=False, hoverinfo="skip"))
                if p["lo"] != p["hi"]:
                    fig.add_trace(go.Scatter(x=[p["lo"], p["hi"]], y=[row, row], mode="lines",
                                             line=dict(color=DARK, width=7), showlegend=False, hoverinfo="skip"))
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

    elif form == "donut":
        # Parts of one whole, largest first from 12 o'clock
        pts.sort(key=lambda p: -p["hi"])
        shades = ["#1D3557", "#457B9D", "#A8DADC", "#D3E9EA"]
        fig.add_trace(go.Pie(
            labels=[p["label"] for p in pts], values=[p["hi"] for p in pts], hole=0.5, sort=False,
            direction="clockwise", marker=dict(colors=shades[:len(pts)], line=dict(color="white", width=2)),
            text=[show(p) for p in pts], texttemplate="%{label}<br>%{text}", textposition="outside",
            hovertext=[p["hover"] for p in pts], hoverinfo="text", showlegend=False))

    elif form == "stacked100":
        # One 100% bar per whole (the series), split into its parts
        wholes = list(dict.fromkeys(p["series"] for p in pts))
        shades = iter(["#1D3557", "#457B9D", "#7FA7C9", "#A8DADC", "#D3E9EA", "#E8E8E8", "#B0B0B0"])
        for part in dict.fromkeys(p["label"] for p in pts):
            sp = {p["series"]: p for p in pts if p["label"] == part}
            fig.add_trace(go.Bar(
                y=[_wrap(w) for w in wholes], x=[sp[w]["hi"] if w in sp else 0 for w in wholes],
                name=_wrap(part, 40), orientation="h", marker_color=next(shades, GRAY),
                text=[show(sp[w]) if w in sp else "" for w in wholes], textposition="inside",
                insidetextanchor="middle", hovertext=[sp[w]["hover"] if w in sp else "" for w in wholes],
                hoverinfo="text"))
        legend = True
        fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 101], title=axis)

    elif form == "area":
        # Parts of a whole over the years, stacked
        shades = iter(["#1D3557", "#457B9D", "#7FA7C9", "#A8DADC", "#D3E9EA", "#B0B0B0"])
        for part in dict.fromkeys(p["label"] for p in pts):
            sp = sorted((p for p in pts if p["label"] == part and p["year"]), key=lambda p: p["year"])
            color = next(shades, GRAY)
            fig.add_trace(go.Scatter(
                x=[int(p["year"]) for p in sp], y=[p["hi"] for p in sp], name=_wrap(part, 40),
                stackgroup="one", mode="lines", line=dict(width=0.5, color=color), fillcolor=color,
                hovertext=[p["hover"] for p in sp], hoverinfo="text"))
        # Each part's value at the first and last year, in the middle of its band
        years = sorted({p["year"] for p in pts if p["year"]})
        for year, side in ((years[0], "left"), (years[-1], "right")):
            base = 0
            for part in dict.fromkeys(p["label"] for p in pts):
                p = next((q for q in pts if q["label"] == part and q["year"] == year), None)
                if p:
                    fig.add_annotation(x=int(year), y=base + p["hi"] / 2, text=show(p), showarrow=False,
                                       xanchor=side, xshift=6 if side == "left" else -6,
                                       font=dict(color="white", size=13))
                    base += p["hi"]
        legend = True
        fig.update_xaxes(title="", dtick=1, tickformat="d")
        fig.update_yaxes(title=axis, range=[0, 100] if share else None)

    elif form == "slope":
        # Two years per category: a line from the first to the second, values at both ends
        years = sorted({p["year"] for p in pts if p["year"]})
        a, b = years[0], years[-1]
        for name in dict.fromkeys(p["label"] for p in pts):
            pair = {p["year"]: p for p in pts if p["label"] == name and p["year"] in (a, b)}
            if len(pair) < 2:
                continue
            up = pair[b]["hi"] >= pair[a]["hi"]
            fig.add_trace(go.Scatter(
                x=[a, b], y=[pair[a]["hi"], pair[b]["hi"]], mode="lines+markers+text",
                line=dict(color=DARK if up else ACCENT, width=2), marker=dict(size=9),
                text=[f"{_wrap(name, 26)}  {show(pair[a])}", f"{show(pair[b])}  {_wrap(name, 26)}"],
                textposition=["middle left", "middle right"], cliponaxis=False, showlegend=False,
                hovertext=[pair[a]["hover"], pair[b]["hover"]], hoverinfo="text"))
        fig.update_xaxes(type="category", title="", range=[-0.9, 1.9])
        fig.update_yaxes(title=axis, showticklabels=False)

    elif form in ("scatter", "bubble", "connected"):
        # Two measures per item; a third as bubble size; an item over the years as a connected path
        ax = f"{c['measure_x']} ({c['unit_x']})" if c.get("unit_x") else c.get("measure_x") or "x"
        if form == "connected":
            palette = iter([DARK, BLUE, ACCENT, MUTED, GRAY])
            for name in dict.fromkeys(p["label"] for p in pts):
                sp = sorted((p for p in pts if p["label"] == name and p["year"]), key=lambda p: p["year"])
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in sp], y=[p["hi"] for p in sp], name=_wrap(name, 40),
                    mode="lines+markers+text", text=[p["year"] for p in sp], textposition="top center",
                    line=dict(color=next(palette, GRAY), width=2), marker=dict(size=9),
                    hovertext=[p["hover"] for p in sp], hoverinfo="text"))
            legend = True
        else:
            size = ([12] * len(pts) if form == "scatter" else
                    [10 + 40 * (p["size"] / max(q["size"] for q in pts)) ** 0.5 for p in pts])
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pts], y=[p["hi"] for p in pts], mode="markers+text",
                text=[_wrap(p["label"], 20) for p in pts], textposition="top center", cliponaxis=False,
                marker=dict(size=size, color=DARK, opacity=0.75, line=dict(color="white", width=1)),
                hovertext=[p["hover"] + (f"<br>{c['measure_size']}: {p['size']:g} {c['unit_size']}"
                                         if p["size"] is not None else "") for p in pts],
                hoverinfo="text", showlegend=False))
        fig.update_xaxes(title=ax, showgrid=True, gridcolor="#EEEEEE")
        fig.update_yaxes(title=axis, showgrid=True, gridcolor="#EEEEEE")

    elif form == "sankey":
        # Amounts from one group (category) to another (series)
        nodes = list(dict.fromkeys([p["label"] for p in pts] + [p["series"] for p in pts]))
        if "%" in unit:
            # Shares of different wholes do not add up: each source lists its own split
            names = [_wrap(n, 30) + "".join(f"<br>→ {p['series']}: {show(p)}" for p in pts if p["label"] == n)
                     for n in nodes]
        else:
            names = [f"{_wrap(n, 30)}: {_fmt(sum(p['hi'] for p in pts if n in (p['label'], p['series'])), unit)}"
                     for n in nodes]
        fig.add_trace(go.Sankey(
            node=dict(label=names, color=DARK, pad=18, thickness=16),
            link=dict(source=[nodes.index(p["label"]) for p in pts], target=[nodes.index(p["series"]) for p in pts],
                      value=[p["hi"] for p in pts], color="rgba(69,123,157,0.35)",
                      customdata=[p["hover"] for p in pts], hovertemplate="%{customdata}<extra></extra>")))

    elif form == "funnel":
        # Each stage a subset of the one before, in the order given
        fig.add_trace(go.Funnel(
            y=[_wrap(p["label"]) for p in pts], x=[p["hi"] for p in pts], marker=dict(color=DARK),
            text=[show(p) for p in pts], textinfo="text+percent initial", textposition="inside",
            hovertext=[p["hover"] for p in pts], hoverinfo="text", connector=dict(fillcolor="#D3E9EA")))

    elif form == "waterfall":
        # Start total, what each change adds or takes away, end total
        kinds = ["absolute" if i == 0 and p["series"].lower() == "total" else
                 "total" if p["series"].lower() == "total" else "relative" for i, p in enumerate(pts)]
        fig.add_trace(go.Waterfall(
            x=[_wrap(p["label"], 16) for p in pts], y=[p["hi"] for p in pts], measure=kinds,
            text=[_fmt(p["hi"], unit) if k != "relative" else show(p) for p, k in zip(pts, kinds)],
            textposition="outside", cliponaxis=False, increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=ACCENT)),
            totals=dict(marker=dict(color=DARK)), connector=dict(line=dict(color=GRAY, width=1)),
            hovertext=[p["hover"] for p in pts], hoverinfo="text"))
        fig.update_yaxes(title=axis)

    elif form == "sunburst":
        # Parts inside their larger groups (series), the groups' size the sum of their parts
        groups = list(dict.fromkeys(p["series"] for p in pts))
        fig.add_trace(go.Sunburst(
            ids=groups + [f"{p['series']}/{p['label']}" for p in pts],
            labels=[_wrap(g, 18) for g in groups] + [f"{_wrap(p['label'], 18)}<br>{show(p)}" for p in pts],
            parents=[""] * len(groups) + [p["series"] for p in pts],
            values=[0] * len(groups) + [p["hi"] for p in pts], branchvalues="remainder",
            marker=dict(colorscale=[[0, "#D3E9EA"], [1, DARK]]),
            hovertext=groups + [p["hover"] for p in pts], hoverinfo="text"))

    elif form == "radar":
        # Up to 3 items across 5+ series of the same measure
        axes = list(dict.fromkeys(p["series"] for p in pts))
        palette = iter([DARK, ACCENT, BLUE])
        for name in dict.fromkeys(p["label"] for p in pts):
            cell = {p["series"]: p for p in pts if p["label"] == name}
            color = next(palette, GRAY)
            fig.add_trace(go.Scatterpolar(
                r=[cell[a]["hi"] for a in axes] + [cell[axes[0]]["hi"]], theta=[_wrap(a, 16) for a in axes] + [_wrap(axes[0], 16)],
                name=_wrap(name, 40), fill="toself", opacity=0.6, line=dict(color=color),
                hovertext=[cell[a]["hover"] for a in axes] + [cell[axes[0]]["hover"]], hoverinfo="text"))
        legend = True
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100] if share else None, tickfont=dict(size=10),
                                                     angle=90, tickangle=90)))

    elif form in ("boxes", "violin"):
        # Several figures per category: their spread, every figure shown
        for name in dict.fromkeys(p["label"] for p in pts):
            sp = [p for p in pts if p["label"] == name]
            kw = dict(x=[(p["lo"] + p["hi"]) / 2 for p in sp], y=[_wrap(name)] * len(sp), orientation="h",
                      name=_wrap(name, 30), line_color=DARK, fillcolor="rgba(168,218,220,0.4)",
                      hovertext=[p["hover"] for p in sp], hoverinfo="text", showlegend=False)
            if form == "violin":
                fig.add_trace(go.Violin(points="all", pointpos=0, jitter=0.4, box_visible=True, **kw))
            else:
                fig.add_trace(go.Box(boxpoints="all", pointpos=0, jitter=0.4, **kw))
        fig.update_yaxes(autorange="reversed")
        value_axis()

    elif form == "histogram":
        # How the figures for one measure are spread
        values = [(p["lo"] + p["hi"]) / 2 for p in pts]
        fig.add_trace(go.Histogram(x=values, nbinsx=max(5, round(len(values) ** 0.5) + 2),
                                   marker=dict(color=DARK, line=dict(color="white", width=1)),
                                   hovertemplate="%{x}: %{y} figures<extra></extra>", showlegend=False))
        fig.update_xaxes(title=axis)
        fig.update_yaxes(title="Number of figures", dtick=1)

    elif form == "bubblemap":
        # Counts per country as circles sized by value
        top = max(p["hi"] for p in pts) or 1
        fig.add_trace(go.Scattergeo(
            locations=[_country(p["label"]) for p in pts], mode="markers+text", text=[show(p) for p in pts],
            textposition="top center", marker=dict(size=[8 + 42 * (p["hi"] / top) ** 0.5 for p in pts],
                                                   color=DARK, opacity=0.7, line=dict(color="white", width=1)),
            hovertext=[p["hover"] for p in pts], hoverinfo="text", showlegend=False))
        fig.update_geos(showframe=False, showland=True, landcolor="#F4F4F4", showcountries=True,
                        countrycolor="#B0B0B0", showcoastlines=True, coastlinecolor="#B0B0B0",
                        projection_type="natural earth", lataxis_range=[-58, 85], lonaxis_range=[-180, 180])

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
    """Break a long label into lines of about `width` characters for the chart axis.
    A word longer than a line (a URL) is broken after a slash."""
    lines, line = [], ""
    for word in text.split():
        joiner = " " if line else ""
        while len(line) + len(joiner) + len(word) > width and "/" in word[:width]:
            cut = word.rindex("/", 0, width) + 1
            if line and len(line) + len(joiner) + cut > width:
                lines.append(line)
                line, joiner = "", ""
                continue
            lines.append(line + joiner + word[:cut])
            line, joiner, word = "", "", word[cut:]
        if line and len(line) + 1 + len(word) > width:
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


def _p_text(p):
    """A p-value as text: "p<0.001" or "p=0.032"."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _verdict(p):
    """What a test result means, in plain words."""
    return (f"a difference unlikely to be chance, {_p_text(p)}" if p < 0.05
            else f"not clearly different from chance, {_p_text(p)}")
