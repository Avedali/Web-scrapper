"""
viz.py: Mode 3, generates charts in two formats.
==================================================
Run: python agent.py viz
Or:  launcher.py → option 6

Generates every chart twice:
  Paper version: information-dense, with axis labels, figure caption,
     and statistical annotations. Goes in the written paper.
  Slide version: minimal, big font, finding as the title, one accent
     color for the point, gray for everything else. Goes in the 5-slide
     presentation.

The chart type follows the data-to-viz.com guide, which picks a chart
from the kind of data being shown:
  - Numeric grouped by categorical → box plot (shows the spread as well as the average)
  - Two categoricals → heatmap (shows which combinations are common)
  - Counts of one categorical → lollipop chart (less ink than bars)
  - Part of a whole → treemap (easier to read than a pie chart)

Two groups of charts:
  Findings (charts_paper/, charts_slide/): what the evidence shows. One
     chart per key finding of the brief, drawn from the figures its sources
     report (e.g. "Young developers lost 13% of jobs while older ones
     gained 6-9%"), then the simulation results.
  Appendix (charts_paper/appendix/, charts_slide/appendix/): how the
     research was done. Sources per theme, sentiment by source type and
     language, language coverage, topics by language, the mix of source types.

All charts use plotly, which produces interactive HTML charts as well as PNG images.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # For loading simulation results and the brief's findings
import sys   # Stop with an error code when a step can't run
import os    # File/folder operations
import re    # Reading the AI's chart designs
import datetime   # This year, to mark forecasts on trend charts

from core import call_ai, source_label   # AI calls; plain names for source types
import verify                             # Checks each charted number against its source



# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_viz(args):
    """Generate all charts from the analysis and simulation results."""

    print("=" * 60)
    print("MODE 3: VISUALIZE (paper + slide formats)")
    print("=" * 60)

    # ── Check dependencies ──────────────────────────────────────────
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    try:
        # plotly.express = simple, high-level chart functions (px.box, px.bar, etc.)
        # plotly.graph_objects = lower-level building blocks (for custom charts like lollipops)
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("pip install plotly")

    # ── Create output directories ───────────────────────────────────
    # Two separate folders: one for paper-format charts, one for slide-format
    paper_root = os.path.join(args.corpus, "charts_paper")
    slide_root = os.path.join(args.corpus, "charts_slide")
    os.makedirs(paper_root, exist_ok=True)   # Create if doesn't exist
    os.makedirs(slide_root, exist_ok=True)

    # ── Load the analyzed corpus ────────────────────────────────────
    csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
    if not os.path.exists(csv_path):
        sys.exit("Run 'python agent.py analyze' first.")

    df = pd.read_csv(csv_path)  # Load into a DataFrame
    # Plain names for the source types ("vendor/primary" → "AI tool makers (GitHub, OpenAI...)")
    if "quality" in df.columns:
        df["source_type"] = df["quality"].map(source_label)
    charts = 0                  # Counts how many charts have been saved

    # ── Color palette ───────────────────────────────────────────────
    # Design rule: a single accent color marks the main point.
    # Everything else is gray or muted, so the audience sees the main
    # point first because it is the only colored element.
    #
    # These are hex color codes (like HTML colors):
    # "#E63946" = a strong red, "#A8DADC" = soft blue, "#B0B0B0" = gray
    ACCENT = "#E63946"     # Strong red: the main point (the thing to notice)
    MUTED = "#A8DADC"      # Soft blue: context (supporting data)
    GRAY = "#B0B0B0"       # Gray: everything else
    DARK = "#1D3557"       # Dark navy: text and strong lines

    # ── Save function (generates both paper and slide versions) ─────
    numbers = {"findings": 0, "appendix": 0}   # Next figure number in each group

    def save(fig, slug, caption, title_slide, appendix=False, keep_legend=False, note=""):
        """
        Save one chart in both formats (paper and slide).

        Parameters:
            fig:          the plotly figure object
            slug:         filename without number or extension ("sentiment_by_type")
            caption:      descriptive caption for the paper ("Figure 3: " is added)
            title_slide:  the finding, used as the slide title
            appendix:     True for charts about the research method; they
                          go in an "appendix" subfolder and are numbered A1, A2...
            keep_legend:  True when the colors carry meaning that isn't written
                          on the chart (stacked bars), so slides keep the legend
            note:         a line of small print under the chart (where the numbers come from)

        'nonlocal charts' lets us modify the 'charts' counter from
        inside this nested function. Without it, Python would create
        a new local variable instead of updating the outer one.
        """
        nonlocal charts
        bottom = (100 if keep_legend else 20) + (30 if note else 0)   # Room under the chart
        if note:
            fig.add_annotation(text=note, showarrow=False, xref="paper", yref="paper",
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
        # Academic style: descriptive title, smaller font, gridlines visible,
        # legend visible, axis labels included. Goes in the paper.
        fig.update_layout(
            title=title_paper,       # Descriptive caption ("Figure 1: ...")
            title_font_size=14,      # Moderate title size
            font_size=11,            # Body text size
            template="plotly_white",  # Clean white template with gridlines
        )
        if note:
            fig.update_layout(margin=dict(b=bottom + 60))   # Axis title plus the note
        # Save as PNG image at 2× resolution (high quality for printing).
        # width/height are in pixels. 900×550 fits well in a paper.
        # The try/except is there because write_image needs the 'kaleido'
        # package to turn plotly charts into static PNG images. Without
        # kaleido (common on ARM Macs), the error would stop the whole
        # viz mode. So the code catches the error, warns the user once,
        # and saves HTML files, which need no extra package.
        try:
            fig.write_image(
                os.path.join(paper_dir, f"{name}.png"),
                scale=2,       # 2× resolution (1800×1100 actual pixels)
                width=900,
                height=550,
            )
        except Exception as e:
            # PNG export failed, so save an HTML copy for the paper too
            fig.write_html(os.path.join(paper_dir, f"{name}.html"))
            if not hasattr(save, "_warned"):
                print(f"   PNG export failed -- saving HTML instead ({type(e).__name__})")
                if "chrome" in str(e).lower():
                    # kaleido 1.x renders through Google Chrome
                    print("      Install Google Chrome, or run: plotly_get_chrome")
                else:
                    print("      pip install kaleido  (for static PNG export)")
                save._warned = True  # Warn on the first failed chart only

        # ── Slide version ───────────────────────────────────────────
        # Presentation style: the title states the finding, big font,
        # no gridlines, no legend, numbers written directly on the chart.
        # The audience reads the title, looks briefly at the shape, and moves on.
        fig.update_layout(
            title=title_slide,         # The finding itself ("Vendor sources skew positive")
            title_font_size=22,        # Big enough to read from the back of the room
            title_x=0.02,             # Align title to the left (not centered)
            font_size=14,              # Larger body text for projection
            plot_bgcolor="white",      # White chart area
            paper_bgcolor="white",     # White around the chart
            xaxis_showgrid=False,      # No gridlines (cleaner)
            yaxis_showgrid=False,
            showlegend=keep_legend,    # Usually no legend: labels are on the chart itself
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
            margin=dict(l=20, r=20, t=90, b=bottom + (60 if note else 0)),
        )
        # Save as PNG (larger, for projection) and HTML (interactive, for zooming).
        # Same kaleido guard as above.
        try:
            fig.write_image(
                os.path.join(slide_dir, f"{name}.png"),
                scale=2, width=1200, height=700,  # Wider for widescreen slides
            )
        except Exception:
            pass  # Already warned above; the HTML file below is still saved

        # HTML version: the audience can hover over data points, zoom in, etc.
        # Needs no kaleido, so this file is always saved.
        fig.write_html(os.path.join(slide_dir, f"{name}.html"))

        charts += 1
        print(f"   {'appendix/' if appendix else ''}{name}")

    # ════════════════════════════════════════════════════════════════
    # FINDINGS CHARTS: what the evidence shows (drawn from the brief)
    # ════════════════════════════════════════════════════════════════
    # These come first: they illustrate the brief's findings. The charts
    # about how the research was done follow, in the appendix folders.
    brief = None   # The brief's themes and findings, if synthesize has run
    brief_path = os.path.join(args.corpus, "brief_findings.json")
    if not os.path.exists(brief_path):
        print("   No brief yet (run synthesize first): skipping the findings charts")
    else:
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)
        with open(os.path.join(args.corpus, "corpus.json"), encoding="utf-8") as f:
            by_idx = {int(src["idx"]): src for src in json.load(f)}

        # ── One chart per key finding ───────────────────────────────
        # Chart 1 shows the figures behind key finding 1, and so on. For
        # example: "Developer use of AI tools rose from 76% to 84% in a year"
        # (a trend), or the productivity gains tool makers report next to
        # those independent studies measured (bars).
        api_key = getattr(args, "key", None) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("   No API key: skipping the findings charts (each needs one AI call)")
        else:
            this_year = datetime.date.today().year
            for c in _theme_charts(brief, by_idx, api_key, args.model):
                pts, unit = c["points"], c["unit"]
                signed = any(p["lo"] < 0 for p in pts)
                show = lambda p: (_fmt(p["lo"], unit, signed) if p["lo"] == p["hi"] else
                                  f"{_fmt(p['lo'], unit, signed)} to {_fmt(p['hi'], unit, signed)}" if signed
                                  else f"{p['lo']:g}-{_fmt(p['hi'], unit)}")
                for p in pts:
                    p["hover"] = f"{p['label']}: {show(p)}<br>{p['source']}"
                fig = go.Figure()
                if c["kind"] == "bars":
                    # Horizontal bars in the AI's order, top to bottom. The bar
                    # the title is about is red, the others gray (or all navy
                    # when none is marked). A range ("35-45%") is a bar from 35 to 45.
                    marked = any(p["highlight"] for p in pts)
                    colors = [ACCENT if p["highlight"] else GRAY if marked else DARK for p in pts]
                    fig.add_trace(go.Bar(
                        y=_unique([_wrap(p["label"]) for p in pts]),
                        x=[p["hi"] - p["lo"] if p["lo"] != p["hi"] else p["hi"] for p in pts],
                        base=[p["lo"] if p["lo"] != p["hi"] else 0 for p in pts],
                        orientation="h", marker_color=colors,
                        text=[show(p) for p in pts], textposition="outside", cliponaxis=False,
                        hovertext=[p["hover"] for p in pts], hoverinfo="text", showlegend=False))
                    fig.update_layout(yaxis_autorange="reversed", bargap=0.35)
                    if signed:
                        fig.add_vline(x=0, line_color=GRAY, line_width=1)   # "No change" line
                    share = re.search(r"%\s*of", unit)   # "% of developers", not "% gain"
                    if share and max(p["hi"] for p in pts) <= 100 and not signed:
                        fig.update_xaxes(range=[0, 108])   # Shares on a 0-100% scale
                    fig.update_xaxes(title=unit)
                else:
                    # One line per series (label), years along the bottom.
                    # Forecasts (years after this one) get an open marker.
                    series = {}
                    for p in pts:
                        series.setdefault(p["label"], []).append(p)
                    palette = iter([ACCENT, DARK, MUTED, GRAY])
                    for name, sp in sorted(series.items(), key=lambda kv: not any(p["highlight"] for p in kv[1])):
                        sp.sort(key=lambda p: p["year"])
                        future = [int(p["year"]) > this_year for p in sp]
                        fig.add_trace(go.Scatter(
                            x=[int(p["year"]) for p in sp], y=[p["hi"] for p in sp], name=name,
                            mode="lines+markers+text", line=dict(color=next(palette, GRAY), width=2),
                            marker=dict(size=10, symbol=["circle-open" if f else "circle" for f in future]),
                            text=[show(p) + (" (forecast)" if f else "") for p, f in zip(sp, future)],
                            textposition="top center", cliponaxis=False,
                            hovertext=[p["hover"] for p in sp], hoverinfo="text"))
                    fig.update_xaxes(title="", dtick=1, tickformat="d")   # Years at true distances
                    if re.search(r"%\s*of", unit):
                        fig.update_yaxes(range=[0, 105])   # Shares on a 0-100% scale
                    fig.update_yaxes(title=unit)
                # Which sources the numbers come from, as the brief cites them ([85])
                cited = ", ".join(f"[{i}]" for i in dict.fromkeys(p["idx"] for p in pts) if i is not None)
                title = c["title"] or c["theme"]
                save(fig, re.sub(r"\W+", "_", c["theme"].lower()).strip("_")[:40],
                     f"{c['theme']}: {title}", _wrap(title, 70),
                     keep_legend=c["kind"] == "trend" and len({p["label"] for p in pts}) > 1,
                     note=f"Sources: {cited} in the brief's references. The figures come from "
                          "different studies and samples.")

    # ════════════════════════════════════════════════════════════════
    # APPENDIX: charts about how the research was done
    # ════════════════════════════════════════════════════════════════

    # ── How many sources each theme of the brief cites ──────────
    # One bar per theme, split by the type of source. This describes the
    # evidence base, so it goes in the appendix with the other method charts.
    themes = [t for t in (brief or {}).get("themes", []) if t.get("sources_by_type")]
    if themes:
        kinds = sorted({k for t in themes for k in t["sources_by_type"]},
                       key=lambda k: -sum(t["sources_by_type"].get(k, 0) for t in themes))
        # A color per type, so the legend can tell them apart. AI tool
        # makers keep the accent red, as in the other charts.
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
    #
    # Why a box plot instead of a bar chart of averages:
    # a bar chart shows one number per group (the average). It hides:
    #   - How spread out the data is within each group
    #   - Whether there are outliers
    #   - Whether the data is skewed
    #
    # A box plot shows all three:
    #   - The box = middle 50% of the data (25th to 75th percentile)
    #   - The line in the box = the median (middle value)
    #   - The whiskers = the range (excluding outliers)
    #   - Individual dots = each data point (points="all")
    #
    # This supports findings such as: "vendor sources are more positive
    # on average and also consistently positive (narrow box), while forum
    # sources vary widely (wide box)."
    #
    # data-to-viz.com: numeric (sentiment) × categorical (quality) → box plot

    if "sentiment" in df.columns and "source_type" in df.columns:

        # Sort categories by median sentiment (lowest to highest)
        # so the chart reads naturally from negative at top to positive at bottom
        order = (
            df.groupby("source_type")["sentiment"]  # Group by type of source
            .median()                            # Get median of each group
            .sort_values()                       # Sort low to high
            .index                               # Get the source types in that order
            .tolist()                            # Convert to a plain list
        )

        # Create a horizontal box plot
        # x = numeric values (sentiment scores)
        # y = categories (quality tiers)
        # category_orders forces the order calculated above
        # color_discrete_sequence sets the box color
        # points="all" shows every individual data point as a dot
        fig = px.box(
            df, x="sentiment", y="source_type",
            category_orders={"source_type": order},
            labels={"source_type": "Type of source", "sentiment": "Sentiment (-1 negative to +1 positive)"},
            color_discrete_sequence=[MUTED],
            points="all",  # Show each source as a dot
        )

        # Add a vertical dashed line at x=0 (the neutral boundary)
        # Everything left of this line is negative, right is positive
        fig.add_vline(x=0, line_dash="dash", line_color=GRAY, line_width=1)

        n_total = len(df)

        # Slide title: the finding itself. It is computed from the data, so
        # the slide can't claim a result the corpus doesn't show.
        # (Plotly titles use <br> for line breaks, not \n.)
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
    #
    # Why a lollipop chart instead of a bar chart:
    # a lollipop chart is a bar chart where each bar is replaced by a
    # thin line and a dot. It uses less ink, and the dot marks the exact
    # value.
    #
    # Each language gets a horizontal line from 0 to its mean sentiment,
    # with a dot at the end showing the value and sample size.
    #
    # Languages with strong sentiment (more than 0.1 away from 0) get the
    # ACCENT color. Languages near 0 get GRAY, so the strong ones stand out.
    #
    # data-to-viz.com: one numeric per category → lollipop

    if "sentiment" in df.columns and "lang" in df.columns:

        # Calculate mean sentiment and count per language.
        # .agg() computes multiple statistics at once:
        # m = mean of sentiment, n = count of sentiment
        data = df.groupby("lang").agg(
            m=("sentiment", "mean"),   # Mean sentiment
            n=("sentiment", "count"),  # Number of sources
        ).reset_index()  # Convert the groupby result back to a normal DataFrame

        # Only keep languages with 2+ sources (can't draw conclusions from 1)
        data = data[data["n"] >= 2].sort_values("m")  # Sort by mean sentiment

        if len(data) >= 2:  # Need at least 2 languages to compare

            # Create an empty figure (traces are added one by one below)
            # A "trace" in plotly is one visual element (a line, a set of dots, etc.)
            fig = go.Figure()

            for _, row in data.iterrows():
                # Highlight languages with strong sentiment (far from 0)
                # abs() = absolute value (ignores the sign: abs(-0.3) = 0.3)
                color = ACCENT if abs(row["m"]) > 0.1 else GRAY

                # The "stick" part of the lollipop: a horizontal line from 0 to the value
                # go.Scatter with mode="lines" draws a line between two points
                fig.add_trace(go.Scatter(
                    x=[0, row["m"]],                         # From 0 to the mean
                    y=[row["lang"], row["lang"]],             # Same y for both points (horizontal)
                    mode="lines",                             # Draw a line between the points
                    line=dict(color=GRAY, width=2),           # Thin gray line
                    showlegend=False,                         # Don't add to legend
                ))

                # The "dot" at the end of the lollipop, with a text label
                fig.add_trace(go.Scatter(
                    x=[row["m"]],                             # Position: the mean sentiment
                    y=[row["lang"]],                          # Position: the language name
                    mode="markers+text",                      # Draw a dot and a text label
                    marker=dict(size=14, color=color),        # Dot size and color
                    # Label shows the value and sample size: "+0.15 (n=12)"
                    text=[f"{row['m']:+.2f} (n={int(row['n'])})"],
                    textposition="middle right",              # Put text to the right of the dot
                    showlegend=False,
                ))

            # Add a solid vertical line at x=0 (the neutral boundary)
            fig.add_vline(x=0, line_dash="solid", line_color=DARK, line_width=1)

            # Leave room right of the dots for the "+0.44 (n=12)" labels
            lo_x, hi_x = min(0, data["m"].min()), max(0, data["m"].max())
            span = (hi_x - lo_x) or 1
            fig.update_xaxes(range=[lo_x - 0.05 * span, hi_x + 0.3 * span])

            # Slide title from the data: the most and least positive languages
            lo, hi = data.iloc[0], data.iloc[-1]
            save(fig, "sentiment_by_language",
                 "Average sentiment of the sources in each language",
                 f"Most positive: {hi['lang']} ({hi['m']:+.2f}). "
                 f"Least positive: {lo['lang']} ({lo['m']:+.2f})", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 3: Language coverage (treemap)
    # ════════════════════════════════════════════════════════════════
    #
    # Why a treemap instead of a pie chart or bar chart:
    # a treemap shows parts of a whole as nested rectangles. Bigger
    # rectangles = more sources in that language. It is easier to read
    # than a pie chart (people compare areas better than angles) and
    # takes less space than a bar chart.
    #
    # data-to-viz.com: counts of one categorical → treemap

    if "lang" in df.columns:

        # Count sources per language
        counts = df["lang"].value_counts().reset_index()
        counts.columns = ["lang", "count"]  # Rename columns for clarity

        # px.treemap draws nested rectangles sized by "values"
        # path=["lang"] means each language gets its own rectangle
        # color="count" colors rectangles by count (more = darker)
        # color_continuous_scale defines the color range: light → dark
        fig = px.treemap(
            counts,
            path=["lang"],       # One rectangle per language
            values="count",      # Size by count
            color="count",       # Color by count
            color_continuous_scale=["#F1FAEE", DARK],  # Light cream to dark navy
        )

        save(fig, "language_coverage",
             f"Number of sources in each language (N={len(df)})",
             f"Sources in {df['lang'].nunique()} languages", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 4: Topic × Language (heatmap)
    # ════════════════════════════════════════════════════════════════
    #
    # Why a heatmap: the chart crosses two categorical variables
    # (language × topic). A heatmap shows the count in each cell as a
    # color: dark = many, light = few. Patterns show up at a glance, for
    # example "Japanese sources cluster in the 'workforce' topic, English
    # in 'productivity'."
    #
    # data-to-viz.com: two categoricals → heatmap/mosaic

    # Prefer analyze's named BERTopic themes; fall back to zero-shot topics
    topic_col = "theme" if "theme" in df.columns else "topic_zs"

    if topic_col in df.columns and "lang" in df.columns:

        # Build a cross-tabulation (contingency table):
        # rows = languages, columns = topics, cells = counts
        # (sources with no usable evidence have no theme, so they are left out)
        sub = df[df[topic_col].notna() & (df[topic_col] != "no usable evidence")]
        ct = pd.crosstab(sub["lang"], sub[topic_col])

        # Need at least 2 languages and 2 topics for a meaningful heatmap
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:

            # Shorten long topic names so they fit on the chart
            # "developer productivity and efficiency" → "developer productivity and eff..."
            ct.columns = [
                c[:30] + "..." if len(c) > 30 else c
                for c in ct.columns
            ]

            # px.imshow renders a 2D array as a heatmap
            # text_auto=True writes the number in each cell
            # "YlOrRd" = Yellow → Orange → Red color scale
            # aspect="auto" lets plotly choose the aspect ratio
            fig = px.imshow(
                ct,
                text_auto=True,                # Show numbers in cells
                color_continuous_scale="YlOrRd",  # Yellow-Orange-Red
                aspect="auto",
            )

            save(fig, "themes_by_language",
                 "Themes discussed by the sources in each language",
                 "Which languages discuss which themes", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHART 5: Source quality breakdown (lollipop)
    # ════════════════════════════════════════════════════════════════
    #
    # A lollipop chart, as in chart 2. Shows how many sources there are
    # in each quality tier. Vendor/SEO sources get the ACCENT color
    # because they may be biased.

    if "source_type" in df.columns:

        # Count sources of each type, sorted smallest to largest
        counts = df["source_type"].value_counts().sort_values()

        fig = go.Figure()

        for q, c in counts.items():
            # Highlight AI tool makers and low-quality sites (the most likely to be biased)
            color = ACCENT if q.startswith(("AI tool makers", "Low-quality")) else MUTED

            # The stick (horizontal line from 0 to the count)
            fig.add_trace(go.Scatter(
                x=[0, c], y=[q, q],
                mode="lines", line=dict(color=GRAY, width=2),
                showlegend=False,
            ))

            # The dot (with the count as a label)
            fig.add_trace(go.Scatter(
                x=[c], y=[q],
                mode="markers+text",
                marker=dict(size=12, color=color),
                text=[str(c)],                    # Show the count number
                textposition="middle right",
                showlegend=False,
            ))

        fig.update_xaxes(range=[0, counts.max() * 1.15])   # Room for the count labels
        save(fig, "source_types",
             f"Number of sources of each type (N={len(df)})",
             f"Where the {len(df)} sources come from", appendix=True)

    # ════════════════════════════════════════════════════════════════
    # CHARTS 6-7: SIMULATION RESULTS (if simulate mode was run)
    # ════════════════════════════════════════════════════════════════
    #
    # These charts only appear if you ran 'python agent.py simulate'
    # before running viz. They visualize:
    #   6. Lines of code by skill level (layman vs junior vs senior)
    #   7. Security guardrail behavior by model

    sim_path = os.path.join(args.corpus, "simulation_results.json")

    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim = json.load(f)

        # ── Chart 6: Skill level → output quality ───────────────────
        # Uses the visual task results (search page, kanban, data explorer)
        # sim.get("visual_results", ...) tries "visual_results" first,
        # falls back to "coding_results" (for compatibility with older runs)
        vr = sim.get("visual_results", sim.get("coding_results", []))

        if vr:
            sim_df = pd.DataFrame(vr)

            # The column name might be "level" (visual tasks) or
            # "prompt_type" (coding tasks) depending on which simulate version ran
            level_col = "level" if "level" in sim_df.columns else "prompt_type"

            if level_col in sim_df.columns and "pro_met" in sim_df.columns:

                # Box plot: share of the professional (senior) spec each page
                # met, by prompt level, as tested in a browser by simulate.
                # This is the experiment's main measure. Lines of code say
                # little about whether a page does what a product needs.
                sim_df = sim_df[sim_df["pro_total"].notna() & (sim_df["pro_total"] > 0)].copy()
                sim_df["spec_pct"] = 100 * sim_df["pro_met"] / sim_df["pro_total"]
                fig = px.box(
                    sim_df, x=level_col, y="spec_pct",
                    color=level_col,
                    color_discrete_map={"layman": GRAY, "junior": MUTED, "senior": ACCENT},
                    category_orders={level_col: ["layman", "junior", "senior"]},
                    points="all",  # Show each page as a dot
                    labels={"spec_pct": "% of a professional's requirements met"},
                )
                fig.update_yaxes(range=[0, 105])

                means = sim_df.groupby(level_col)["spec_pct"].mean()
                slide_title = (f"Prompts written like a senior developer met {means.get('senior', 0):.0f}% "
                               f"of a professional's requirements.<br>A layman's prompts met "
                               f"{means.get('layman', 0):.0f}%")
                save(fig, "prompt_skill_vs_result",
                     "Share of a professional's requirements each AI-built page met, by how "
                     "skilled the prompt was (each page tested in a browser)",
                     slide_title)

            elif level_col in sim_df.columns:

                # Older simulate results (no browser test): code length instead
                # Box plot: code length (lines) by skill level
                # color_discrete_map assigns specific colors to each level:
                #   layman = gray (baseline), senior = red (the main point)
                fig = px.box(
                    sim_df, x=level_col, y="lines",
                    color=level_col,
                    color_discrete_map={
                        "layman": GRAY,    # Layman: gray (baseline)
                        "junior": MUTED,   # Junior: blue (context)
                        "senior": ACCENT,  # Senior: red (the finding)
                    },
                    # Force the order: layman → junior → senior (left to right)
                    category_orders={level_col: ["layman", "junior", "senior"]},
                    points="all",  # Show each individual result as a dot
                )

                # Slide title from the data: senior vs layman median length
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
        # Shows which models write secure code for each security task
        gr = sim.get("guardrail_results", [])

        if gr:
            gr_df = pd.DataFrame(gr)

            # Convert text grades to numbers for plotting
            # SECURE=3 (best), PARTIAL=2, INSECURE=1 (worst)
            grade_map = {"SECURE": 3, "PARTIAL": 2, "INSECURE": 1}
            # .map() replaces each value using the dictionary
            gr_df["score"] = gr_df["grade"].map(grade_map)

            # Grouped bar chart: one group per model, one bar per security task
            # text="grade" writes the text grade (SECURE/PARTIAL/INSECURE) on each bar
            fig = px.bar(
                gr_df, x="model", y="score",
                color="task_name",      # Different color for each security test
                barmode="group",        # Bars side by side (not stacked)
                text="grade",           # Show the text grade on each bar
                color_discrete_sequence=[ACCENT, MUTED, DARK],
            )

            # Replace numeric y-axis labels with text labels
            # tickvals = where to put the labels (1, 2, 3)
            # ticktext = what text to show at each position
            fig.update_yaxes(
                tickvals=[1, 2, 3],
                ticktext=["INSECURE", "PARTIAL", "SECURE"],
            )

            save(fig, "security_tests",
                 "How each AI model handled 3 requests for insecure code",
                 "Not all AI models warn about insecure code", keep_legend=True)

    # ── Note about BERTopic charts ──────────────────────────────────
    # BERTopic generates its own plotly charts during analyze mode.
    # They are already saved in scratchpad/charts/. This step only prints
    # a reminder of where they are.
    bt_dir = os.path.join(args.corpus, "charts")
    existing_bt = (
        [f for f in os.listdir(bt_dir) if f.startswith("bertopic")]
        if os.path.exists(bt_dir) else []
    )
    if existing_bt:
        print(f"   BERTopic charts already in {bt_dir}/: {', '.join(existing_bt)}")

    # ── Done ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{charts} charts generated in two formats:")
    print(f"   Paper:  {paper_root}/ (captioned, axis labels)")
    print(f"   Slides: {slide_root}/ (minimal, finding as title)")
    print("   Charts about the research method are in the appendix/ subfolders.")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# ONE CHART PER KEY FINDING (the findings charts)
# ════════════════════════════════════════════════════════════════════════

MAX_NUMBERED = 80     # Findings with numbers shown to the AI for one theme
MAX_BARS = 8          # Bars (or points) in one chart


def _theme_charts(brief, by_idx, api_key, model):
    """
    For each theme of the brief, ask the AI which of the theme's figures show
    its key finding, and how: as bars (figures side by side, e.g. "69% of
    CEOs say AI is changing their business" next to "1% call their AI use
    mature") or as a trend (the same figure in different years, e.g.
    developers using AI tools: 76% in 2024, 84% in 2025).

    Code then checks every number: a value must appear in its finding and in
    one of that finding's sources, a year must appear in the finding, and
    the title may only contain numbers that are on the chart. Values that
    fail are left out; a title that fails is replaced by the theme's name.

    Returns [{"theme", "kind" ("bars"/"trend"), "title", "unit",
    "points": [{"label", "lo", "hi", "year", "highlight", "idx", "source"}]}].
    """
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
            "Below are this theme's findings that contain numbers (ID: claim). Design ONE chart "
            "that shows a reader what this theme found, using these numbers. Choose:\n"
            "- bars: figures side by side. Every bar is the same kind of number in the same unit "
            "(e.g. all '% of developers' or all '% change in productivity'), and the contrast "
            "between bars is the point (e.g. what tool makers claim vs what independent studies "
            "measured; young vs older developers; what executives expect vs what they have done).\n"
            "- trend: the same figure in different years (e.g. share of developers using AI tools "
            "in 2024 and 2025; a forecast for 2028 counts too).\n"
            "- none: if no 2 numbers here can honestly go on one chart.\n"
            "The chart must be about the subject of the research (what AI is doing to "
            "productivity, jobs, companies, code...), never about the sources themselves.\n\n"
            "Format, exactly:\n"
            "CHART: bars | trend | none\n"
            "TITLE: <what the chart shows, as a plain sentence a non-specialist understands, at "
            "most 14 words, using only numbers from the chart, e.g. 'Young developers lost jobs "
            "while older ones gained'>\n"
            "UNIT: <e.g. % of developers>\n"
            f"Then 2 to {MAX_BARS} lines, in the order to show them:\n"
            "- C12 | <label: who or what this number is about, and who measured it, at most 7 "
            "plain words, e.g. 'Developers aged 22-25 (Stanford)'> | <the number as written in "
            "the finding, e.g. 55 or 35-45; negative for a decrease> | <year, for a trend; "
            "otherwise empty> | <* if this is the number the title is about, otherwise empty>\n\n"
            "FINDINGS:\n" + "\n".join(f"{i}: {f['claim']}" for i, f in found.items()),
            api_key, model, temp=0) or ""
        chart = _read_chart(reply, found, by_idx)
        if not chart:
            print(f"   {t['title']}: no chart (the AI found no comparable numbers)")
            continue
        chart["theme"] = t["title"]
        charts.append(chart)
    return charts


def _read_chart(reply, found, by_idx):
    """Read one chart from the AI's reply and check its numbers (see _theme_charts)."""
    def field(name):
        m = re.search(rf"^\W*{name}\W*:\s*(.+)$", reply, re.I | re.M)
        return m.group(1).strip().strip("*").strip() if m else ""

    kind = field("CHART").lower()
    kind = "trend" if "trend" in kind else "bars" if "bar" in kind else ""
    if not kind:
        return None
    points = []
    for line in reply.splitlines():
        parts = [x.strip() for x in line.strip().lstrip("-• ").split("|")]
        if len(parts) < 3 or parts[0] not in found:
            continue
        f = found[parts[0]]
        m = re.fullmatch(r"([-+−]?)\s*(\d[\d.,]*)\s*%?(?:\s*(?:-|–|to)\s*(\d[\d.,]*)\s*%?)?", parts[2])
        if not m:
            continue
        sources = [by_idx[s] for s in f["sources"][:3] if s in by_idx]
        raws = [r for r in (m.group(2), m.group(3)) if r]
        # The check: each number must be in the finding and in one of its sources
        if not all(_in_text(r, f["claim"]) and any(_in_text(r, s.get("content", "")) for s in sources)
                   for r in raws):
            print(f"   left out of the chart (number not found in its source): {parts[0]} {parts[2]}")
            continue
        sign = -1 if m.group(1) in ("-", "−") else 1
        values = sorted(sign * float(_readings_one(r)) for r in raws)
        year = parts[3] if len(parts) > 3 else ""
        year = year if re.fullmatch(r"(19|20)\d\d", year) and year in f["claim"] else ""
        if kind == "trend" and (not year or len(raws) > 1):
            continue   # A trend point needs a year written in its finding, and one value
        lead = sources[0] if sources else {}
        points.append({
            "label": parts[1][:60] or parts[0],
            "lo": values[0], "hi": values[-1], "year": year,
            "highlight": len(parts) > 4 and "*" in parts[4],
            "idx": lead.get("idx"),
            "source": f"{lead.get('title', '')[:80]} ({lead.get('domain', '')})",
        })
    points = points[:MAX_BARS]
    if len(points) < 2:
        return None

    # The title may only use numbers that are on the chart
    title = field("TITLE")
    allowed = set()
    for p in points:
        for v in (p["lo"], p["hi"]):
            allowed |= verify._readings(f"{abs(v):g}") | {f"{round(abs(v)):g}"}   # 54.9 may be written 55
        allowed.add(p["year"])
    unknown = {n for n in re.findall(r"\d+(?:[.,]\d+)?", title)
               if not (verify._readings(n.replace(",", ".")) & allowed)}
    if unknown or not title:
        if title:
            print(f"   chart title replaced (numbers not on the chart: {', '.join(sorted(unknown))}): {title}")
        title = ""
    return {"kind": kind, "title": title, "unit": field("UNIT"), "points": points}


def _in_text(raw, text):
    """True if the number `raw` ("55", "1", "0,47") is written in `text`."""
    if verify._readings(raw.replace(",", ".")) & verify.numbers(text):
        return True
    # numbers() skips single digits ("only 1% call it mature"), so look for those directly
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
    """
    A figure as text: 84 with unit "%" → "84%"; with signed=True (a measure
    of change) → "+84%"; 3.5 with unit "hours" → "3.5 hours".
    """
    number = f"{value:+g}" if signed else f"{value:g}"
    return f"{number}%" if "%" in unit else f"{number} {unit}".strip()


def _unique(labels):
    """Make chart labels unique: a second "GitHub survey" becomes "GitHub survey (2)"."""
    seen, out = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out
