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

All charts use plotly, which produces interactive HTML charts as well as PNG images.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # For loading simulation results
import sys   # Stop with an error code when a step can't run
import os    # File/folder operations



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
    paper_dir = os.path.join(args.corpus, "charts_paper")
    slide_dir = os.path.join(args.corpus, "charts_slide")
    os.makedirs(paper_dir, exist_ok=True)   # Create if doesn't exist
    os.makedirs(slide_dir, exist_ok=True)

    # ── Load the analyzed corpus ────────────────────────────────────
    csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
    if not os.path.exists(csv_path):
        sys.exit("Run 'python agent.py analyze' first.")

    df = pd.read_csv(csv_path)  # Load into a DataFrame
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
    def save(fig, name, title_paper, title_slide):
        """
        Save one chart in both formats (paper and slide).

        Parameters:
            fig:          the plotly figure object
            name:         filename (without extension)
            title_paper:  descriptive caption for the paper
            title_slide:  the finding, used as the slide title

        'nonlocal charts' lets us modify the 'charts' counter from
        inside this nested function. Without it, Python would create
        a new local variable instead of updating the outer one.
        """
        nonlocal charts

        # ── Paper version ───────────────────────────────────────────
        # Academic style: descriptive title, smaller font, gridlines visible,
        # legend visible, axis labels included. Goes in the paper.
        fig.update_layout(
            title=title_paper,       # Descriptive caption ("Figure 1: ...")
            title_font_size=14,      # Moderate title size
            font_size=11,            # Body text size
            template="plotly_white",  # Clean white template with gridlines
        )
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
            showlegend=False,          # No legend (labels are on the chart itself)
            margin=dict(l=20, r=20, t=60, b=20),  # Tight margins (more chart, less whitespace)
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
        print(f"   {name}")

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

    if "sentiment" in df.columns and "quality" in df.columns:

        # Sort categories by median sentiment (lowest to highest)
        # so the chart reads naturally from negative at top to positive at bottom
        order = (
            df.groupby("quality")["sentiment"]  # Group by quality tier
            .median()                            # Get median of each group
            .sort_values()                       # Sort low to high
            .index                               # Get the quality tier names in that order
            .tolist()                            # Convert to a plain list
        )

        # Create a horizontal box plot
        # x = numeric values (sentiment scores)
        # y = categories (quality tiers)
        # category_orders forces the order calculated above
        # color_discrete_sequence sets the box color
        # points="all" shows every individual data point as a dot
        fig = px.box(
            df, x="sentiment", y="quality",
            category_orders={"quality": order},
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
            slide_title = (f"Vendor sources average {v_mean:+.2f} sentiment --<br>"
                           f"other sources average {o_mean:+.2f}")
        else:
            slide_title = "Sentiment by source type"

        save(fig, "01_sentiment_by_type",
             # Paper title: descriptive, includes N
             f"Figure 1: Sentiment distribution by source quality tier (N={n_total})",
             slide_title)

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
            save(fig, "02_sentiment_by_lang",
                 "Figure 2: Mean sentiment by source language (lollipop chart)",
                 f"Most positive: {hi['lang']} ({hi['m']:+.2f}) -- "
                 f"least positive: {lo['lang']} ({lo['m']:+.2f})")

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

        save(fig, "03_language_coverage",
             f"Figure 3: Source distribution by language (N={len(df)})",
             f"{df['lang'].nunique()} languages -- not just English")

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

            save(fig, "04_topic_language",
                 "Figure 4: Theme distribution by source language",
                 "Which languages discuss which themes")

    # ════════════════════════════════════════════════════════════════
    # CHART 5: Source quality breakdown (lollipop)
    # ════════════════════════════════════════════════════════════════
    #
    # A lollipop chart, as in chart 2. Shows how many sources there are
    # in each quality tier. Vendor/SEO sources get the ACCENT color
    # because they may be biased.

    if "quality" in df.columns:

        # Count sources per quality tier, sorted smallest to largest
        counts = df["quality"].value_counts().sort_values()

        fig = go.Figure()

        for q, c in counts.items():
            # Highlight vendor and SEO sources (the ones most likely to be biased)
            color = ACCENT if "vendor" in q.lower() or "SEO" in q else MUTED

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
        save(fig, "05_source_quality",
             f"Figure 5: Source inventory by quality tier (N={len(df)})",
             f"{len(df)} sources -- where did the evidence come from?")

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
                    labels={"spec_pct": "% of professional spec met"},
                )
                fig.update_yaxes(range=[0, 105])

                means = sim_df.groupby(level_col)["spec_pct"].mean()
                slide_title = (f"Senior prompts met {means.get('senior', 0):.0f}% of a professional "
                               f"spec -- layman prompts {means.get('layman', 0):.0f}%")
                save(fig, "06_skill_vs_output",
                     "Figure 6: Share of professional requirements met, by prompt sophistication "
                     "(each page tested in a browser)",
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

                save(fig, "06_skill_vs_output",
                     "Figure 6: Code output length by prompt sophistication",
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

            save(fig, "07_guardrails",
                 "Figure 7: Security guardrail behavior by model",
                 "Not all models warn about insecure code")

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
    print(f"   Paper:  {paper_dir}/ (captioned, axis labels)")
    print(f"   Slides: {slide_dir}/ (minimal, finding as title)")
    print(f"{'='*60}")
