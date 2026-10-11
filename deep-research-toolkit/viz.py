"""
viz.py -- Mode 3: Generate charts in two formats.
==================================================
Run: python agent.py viz
Or:  launcher.py → option 6

Generates every chart TWICE:
  Paper version: information-dense, with axis labels, figure caption,
     and statistical annotations. Goes in the written paper.
  Slide version: minimal, big font, finding as the title, one accent
     color for the point, gray for everything else. Goes in the 5-slide
     presentation.

Chart type selection follows data-to-viz.com logic -- we pick the chart
type based on what KIND of data we have, not based on habit:
  - Numeric grouped by categorical → box plot (shows spread, not just averages)
  - Two categoricals → heatmap (shows which combinations are common)
  - Counts of one categorical → lollipop chart (cleaner than bars)
  - Part of a whole → treemap (more intuitive than pie chart)

Uses plotly (interactive, polished) for all charts.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # For loading simulation results
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
        print("pip install pandas")
        return

    try:
        # plotly.express = simple, high-level chart functions (px.box, px.bar, etc.)
        # plotly.graph_objects = lower-level building blocks (for custom charts like lollipops)
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("pip install plotly")
        return

    # ── Create output directories ───────────────────────────────────
    # Two separate folders: one for paper-format charts, one for slide-format
    paper_dir = os.path.join(args.corpus, "charts_paper")
    slide_dir = os.path.join(args.corpus, "charts_slide")
    os.makedirs(paper_dir, exist_ok=True)   # Create if doesn't exist
    os.makedirs(slide_dir, exist_ok=True)

    # ── Load the analyzed corpus ────────────────────────────────────
    csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
    if not os.path.exists(csv_path):
        print("Run 'python agent.py analyze' first.")
        return

    df = pd.read_csv(csv_path)  # Load into a DataFrame
    charts = 0                  # Counter for how many charts we've made

    # ── Color palette ───────────────────────────────────────────────
    # Design principle: ONE accent color draws the eye to the point.
    # Everything else is gray or muted. The audience sees the important
    # thing first because it's the only thing that's not gray.
    #
    # These are hex color codes (like HTML colors):
    # "#E63946" = a strong red, "#A8DADC" = soft blue, "#B0B0B0" = gray
    ACCENT = "#E63946"     # Strong red -- for THE point (the thing to notice)
    MUTED = "#A8DADC"      # Soft blue -- for context (supporting data)
    GRAY = "#B0B0B0"       # Gray -- for everything non-focal
    DARK = "#1D3557"       # Dark navy -- for text and strong lines

    # ── Save function (generates both paper and slide versions) ─────
    def save(fig, name, title_paper, title_slide):
        """
        Save one chart in BOTH formats.

        Parameters:
            fig          -- the plotly figure object
            name         -- filename (without extension)
            title_paper  -- descriptive caption for the paper
            title_slide  -- finding-as-title for the slide

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
        # width/height in pixels -- 900×550 fits well in a paper.
        # WRAPPED IN TRY/EXCEPT: write_image needs the 'kaleido' package
        # to convert plotly charts to static PNG images. If kaleido isn't
        # installed (common on ARM Macs), this would crash the entire
        # viz mode. Instead, we catch the error, warn the user, and
        # continue with just HTML output (which always works).
        try:
            fig.write_image(
                os.path.join(paper_dir, f"{name}.png"),
                scale=2,       # 2× resolution (1800×1100 actual pixels)
                width=900,
                height=550,
            )
        except Exception:
            # kaleido not installed or broken -- save HTML fallback for paper too
            fig.write_html(os.path.join(paper_dir, f"{name}.html"))
            if not hasattr(save, "_warned"):
                print("   kaleido not installed -- saving HTML instead of PNG")
                print("      pip install kaleido  (for static PNG export)")
                save._warned = True  # Only warn once, not for every chart

        # ── Slide version ───────────────────────────────────────────
        # Presentation style: the title IS the finding, big font,
        # no gridlines, no legend, numbers annotated directly on the chart.
        # The audience reads the title, glances at the shape, moves on.
        fig.update_layout(
            title=title_slide,         # The finding itself ("Vendor sources skew positive")
            title_font_size=22,        # Big -- readable from the back of the room
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
            pass  # Already warned above; HTML below always works

        # HTML version: audience can hover over data points, zoom in, etc.
        # This ALWAYS works (no kaleido needed).
        fig.write_html(os.path.join(slide_dir, f"{name}.html"))

        charts += 1
        print(f"   {name}")

    # ════════════════════════════════════════════════════════════════
    # CHART 1: Sentiment by source type (BOX PLOT)
    # ════════════════════════════════════════════════════════════════
    #
    # WHY A BOX PLOT (not a bar chart of averages)?
    # A bar chart shows one number per group (the average). It hides:
    #   - How spread out the data is within each group
    #   - Whether there are outliers
    #   - Whether the data is skewed
    #
    # A box plot shows ALL of this:
    #   - The box = middle 50% of the data (25th to 75th percentile)
    #   - The line in the box = the median (middle value)
    #   - The whiskers = the range (excluding outliers)
    #   - Individual dots = each data point (points="all")
    #
    # This tells a richer story: "vendor sources are not just more
    # positive ON AVERAGE -- they're CONSISTENTLY positive (tight box),
    # while forum sources are all over the place (wide box)."
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
        # category_orders forces the order we calculated above
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
        save(fig, "01_sentiment_by_type",
             # Paper title: descriptive, includes N
             f"Figure 1: Sentiment distribution by source quality tier (N={n_total})",
             # Slide title: THE FINDING itself -- what the audience should remember
             "Vendor sources are consistently positive --\nindependent sources tell a different story")

    # ════════════════════════════════════════════════════════════════
    # CHART 2: Sentiment by language (LOLLIPOP CHART)
    # ════════════════════════════════════════════════════════════════
    #
    # WHY A LOLLIPOP (not a bar chart)?
    # A lollipop chart is a bar chart where the bar is replaced by a
    # thin line + dot. It looks cleaner and the dot draws the eye to
    # the actual value. Less visual clutter.
    #
    # Each language gets a horizontal line from 0 to its mean sentiment,
    # with a dot at the end showing the value and sample size.
    #
    # Languages with strong sentiment (far from 0) get the ACCENT color.
    # Languages near 0 get GRAY. This draws attention to the interesting ones.
    #
    # data-to-viz.com: one numeric per category → lollipop

    if "sentiment" in df.columns and "lang" in df.columns:

        # Calculate mean sentiment and count per language.
        # .agg() lets us compute multiple statistics at once:
        # m = mean of sentiment, n = count of sentiment
        data = df.groupby("lang").agg(
            m=("sentiment", "mean"),   # Mean sentiment
            n=("sentiment", "count"),  # Number of sources
        ).reset_index()  # Convert the groupby result back to a normal DataFrame

        # Only keep languages with 2+ sources (can't draw conclusions from 1)
        data = data[data["n"] >= 2].sort_values("m")  # Sort by mean sentiment

        if len(data) >= 2:  # Need at least 2 languages to compare

            # Create an empty figure (we'll add traces manually)
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
                    mode="markers+text",                      # Draw a dot AND text
                    marker=dict(size=14, color=color),        # Dot size and color
                    # Label shows the value and sample size: "+0.15 (n=12)"
                    text=[f"{row['m']:+.2f} (n={int(row['n'])})"],
                    textposition="middle right",              # Put text to the right of the dot
                    showlegend=False,
                ))

            # Add a solid vertical line at x=0 (the neutral boundary)
            fig.add_vline(x=0, line_dash="solid", line_color=DARK, line_width=1)

            save(fig, "02_sentiment_by_lang",
                 "Figure 2: Mean sentiment by source language (lollipop chart)",
                 "Non-English sources see the topic differently")

    # ════════════════════════════════════════════════════════════════
    # CHART 3: Language coverage (TREEMAP)
    # ════════════════════════════════════════════════════════════════
    #
    # WHY A TREEMAP (not a pie chart or bar chart)?
    # A treemap shows part-of-a-whole as nested rectangles. Bigger
    # rectangles = more sources in that language. It's more intuitive
    # than a pie chart (humans are bad at comparing angles) and more
    # compact than a bar chart.
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
    # CHART 4: Topic × Language (HEATMAP)
    # ════════════════════════════════════════════════════════════════
    #
    # WHY A HEATMAP?
    # Two categorical variables (language × topic) crossed together.
    # A heatmap shows the count in each cell as a color: dark = many,
    # light = few. You can instantly see patterns: "Japanese sources
    # cluster in the 'workforce' topic, English in 'productivity'."
    #
    # data-to-viz.com: two categoricals → heatmap/mosaic

    if "topic_zs" in df.columns and "lang" in df.columns:

        # Build a cross-tabulation (contingency table):
        # rows = languages, columns = topics, cells = counts
        ct = pd.crosstab(df["lang"], df["topic_zs"])

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
                 "Figure 4: Topic distribution by source language",
                 "Different languages discuss different concerns")

    # ════════════════════════════════════════════════════════════════
    # CHART 5: Source quality breakdown (LOLLIPOP)
    # ════════════════════════════════════════════════════════════════
    #
    # Same idea as chart 2: lollipop instead of bar. Shows how many
    # sources we have in each quality tier. Vendor/SEO sources get
    # the ACCENT color to draw attention to potential bias.

    if "quality" in df.columns:

        # Count sources per quality tier, sorted smallest to largest
        counts = df["quality"].value_counts().sort_values()

        fig = go.Figure()

        for q, c in counts.items():
            # Highlight vendor and SEO sources (these are the ones to watch out for)
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

            if level_col in sim_df.columns:

                # Box plot: code length (lines) by skill level
                # color_discrete_map assigns specific colors to each level:
                #   layman = gray (not interesting), senior = red (the point)
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

                save(fig, "06_skill_vs_output",
                     "Figure 6: Code output length by prompt sophistication",
                     "Senior prompts get 3-5× more code -- the tool amplifies skill")

        # ── Chart 7: Guardrail behavior ─────────────────────────────
        # Shows which models handle security risks properly
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
    # They're already saved in scratchpad/charts/. We just remind
    # the user they're there.
    bt_dir = os.path.join(args.corpus, "charts")
    existing_bt = (
        [f for f in os.listdir(bt_dir) if f.startswith("bertopic")]
        if os.path.exists(bt_dir) else []
    )
    if existing_bt:
        print(f"   BERTopic charts already in {bt_dir}/: {', '.join(existing_bt)}")

    # ── Done! ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{charts} charts generated in two formats:")
    print(f"   Paper:  {paper_dir}/ (captioned, axis labels)")
    print(f"   Slides: {slide_dir}/ (minimal, finding as title)")
    print(f"{'='*60}")
