"""
stats.py -- Mode 2b: Statistical tests.
========================================
Run: python agent.py stats
Or:  launcher.py → option 3

This mode runs formal statistical tests on the data that analyze
mode produced. It answers questions like:
  - "Are vendor sources SIGNIFICANTLY more positive than forum sources?"
    (not just 'they look more positive' -- is the difference real?)
  - "Does sentiment vary across languages, or is the variation just noise?"
  - "Is there a causal chain: source type → topic → sentiment?"

Two paths depending on your data:
  1. NO external data (default): tests run on the corpus analysis results
     (the CSV that analyze mode saved). Tests: Mann-Whitney, Kruskal-Wallis,
     Spearman, chi-squared, SEM (structural equation modeling).
  2. WITH external data (--data survey.csv): auto-detects columns and runs
     logistic regression on any binary AI-related variable it finds.

Uses pingouin if installed (effect sizes, power, Bayes Factor in one call).
Falls back to scipy if not.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import sys   # Stop with an error code when a step can't run
import os    # File/folder operations

from core import pad  # Scratchpad saving


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION -- decides which path to take
# ════════════════════════════════════════════════════════════════════════

def run_stats(args):
    """
    Entry point for stats mode. Decides whether to analyze the
    corpus data or an external survey CSV.
    """

    print("=" * 60)
    print("MODE 2b: STATISTICAL ANALYSIS")
    print("=" * 60)

    # Check that pandas is available
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    # ── Decision: external survey data or corpus data? ──────────────
    if args.data and os.path.exists(args.data):
        # User passed --data survey.csv → analyze that CSV
        print(f"   External data: {args.data}")
        df = pd.read_csv(args.data)   # Load the CSV into a DataFrame
        _survey_stats(df, args)        # Run survey-specific tests
    else:
        # No external data → use the analyzed corpus from analyze mode
        csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)  # Load the analyzed corpus
            _corpus_stats(df, args)      # Run corpus-specific tests
        else:
            print("Run 'python agent.py analyze' first, or pass --data file.csv")


# ════════════════════════════════════════════════════════════════════════
# PATH 1: CORPUS STATS (testing our own research data)
# ════════════════════════════════════════════════════════════════════════

def _corpus_stats(df, args):
    """
    Run statistical tests on the research corpus.

    The corpus has columns like: sentiment, sent_label, quality, lang,
    topic_zs, bertopic, etc. (added by analyze mode).

    We test relationships between these variables.
    """
    import pandas as pd

    report = "# Statistical Analysis\n\n"  # Will be saved to scratchpad

    # ── Try to load pingouin ────────────────────────────────────────
    # pingouin is the upgrade over raw scipy. Where scipy's ttest gives
    # you just T and p, pingouin gives you T, p, degrees of freedom,
    # Cohen's d (effect size), 95% CI, statistical power, AND Bayes Factor.
    # All in one neat DataFrame. One function call.
    #
    # If pingouin isn't installed, we fall back to scipy (still works,
    # just gives less information per test).
    try:
        import pingouin as pg
        HAS_PG = True
        print("   Using pingouin (effect sizes + Bayes Factor included)")
    except ImportError:
        HAS_PG = False
        print("   pip install pingouin (falling back to scipy)")

    # scipy.stats is the standard Python statistics library
    try:
        from scipy import stats as sp
    except ImportError:
        sys.exit("pip install scipy")

    # ════════════════════════════════════════════════════════════════
    # TEST 0: DESCRIPTIVE STATISTICS
    # ════════════════════════════════════════════════════════════════
    # Before any hypothesis tests, describe the data.
    # "Descriptive" = just summarizing what you have.
    # "Inferential" = testing whether patterns are real or just noise.
    #
    # The 95% confidence interval (CI) means: "if we repeated this
    # study 100 times, the true mean would fall within this range
    # 95 times." It's how sure we are about our estimate.

    if "sentiment" in df.columns:
        print(f"\n{'─'*40}\nDESCRIPTIVE\n{'─'*40}")

        n = len(df)                          # Number of sources
        m = df["sentiment"].mean()           # Average sentiment
        # Standard error = standard deviation / square root of sample size.
        # It measures how precise our estimate of the mean is.
        # Larger sample → smaller SE → more precise estimate.
        se = df["sentiment"].std() / (n ** 0.5)  # n ** 0.5 = √n

        # 95% CI = mean ± 1.96 × standard error
        # 1.96 comes from the normal distribution -- it's the z-score
        # that captures 95% of the area under the bell curve.
        print(f"   Mean sentiment: {m:+.3f} (95% CI: [{m-1.96*se:.3f}, {m+1.96*se:.3f}])")
        print(f"   Std: {df['sentiment'].std():.3f}, n={n}")

        report += (f"## Descriptive\n"
                   f"Mean: {m:+.3f}, 95% CI: [{m-1.96*se:.3f}, {m+1.96*se:.3f}], n={n}\n\n")

    # ════════════════════════════════════════════════════════════════
    # TEST 1: MANN-WHITNEY U (vendor vs non-vendor sentiment)
    # ════════════════════════════════════════════════════════════════
    #
    # WHAT IT TESTS: Are two groups different?
    #   Group 1: sentiment scores from vendor/primary sources
    #   Group 2: sentiment scores from everything else
    #
    # WHY NOT a t-test? The t-test assumes data is normally distributed
    # (bell-curve shaped). Sentiment scores rarely are -- they cluster
    # at the extremes (very positive or very negative). Mann-Whitney
    # doesn't assume any distribution shape (it's "non-parametric").
    #
    # HOW IT WORKS: Rank all scores from lowest to highest, then check
    # if one group's ranks are systematically higher than the other's.
    #
    # THE QUESTION: "If I randomly pick one vendor source and one non-vendor
    # source, what's the probability the vendor source is more positive?"

    if "quality" in df.columns and "sentiment" in df.columns:
        print(f"\n{'─'*40}\nMANN-WHITNEY U: Vendor vs Non-Vendor\n{'─'*40}")

        # Split the data into two groups.
        # df["quality"].str.contains("vendor", case=False, na=False)
        # creates a True/False column: True for rows where quality contains "vendor"
        # case=False = case-insensitive, na=False = treat missing values as False
        vendor = df[df["quality"].str.contains("vendor", case=False, na=False)]["sentiment"]

        # ~ is the NOT operator for pandas: True→False, False→True
        # So this selects everything that's NOT a vendor source
        non_vendor = df[~df["quality"].str.contains("vendor", case=False, na=False)]["sentiment"]

        # Need at least 3 sources in each group for a meaningful test
        if len(vendor) >= 3 and len(non_vendor) >= 3:
            if HAS_PG:
                # pingouin's mwu() returns a DataFrame with:
                #   U-val, p-val, effect size (r), and more
                result = pg.mwu(vendor, non_vendor)
                print(result.to_string())
                report += f"## Mann-Whitney U\n{result.to_string()}\n\n"
            else:
                # scipy's mannwhitneyu returns just the U statistic and p-value
                # alternative="two-sided" means we're testing "is there ANY
                # difference?" (not just "is group A higher than B?")
                u, p = sp.mannwhitneyu(vendor, non_vendor, alternative="two-sided")
                print(f"   U = {u:.1f}, p = {p:.4f}")
                print(f"   Vendor: {vendor.mean():+.3f} (n={len(vendor)})")
                print(f"   Non-vendor: {non_vendor.mean():+.3f} (n={len(non_vendor)})")
                if p < 0.05:
                    print("   Significant")
                else:
                    print("   Not significant")
                report += f"## Mann-Whitney U\nU={u:.1f}, p={p:.4f}\n\n"

    # ════════════════════════════════════════════════════════════════
    # TEST 2: KRUSKAL-WALLIS (sentiment across 3+ languages)
    # ════════════════════════════════════════════════════════════════
    #
    # WHAT IT TESTS: Are THREE OR MORE groups different?
    #   Like Mann-Whitney, but for more than 2 groups.
    #   "Does sentiment differ across English, French, Japanese, etc.?"
    #
    # WHY THIS TEST: It's the non-parametric version of one-way ANOVA.
    #   - ANOVA assumes normal distribution → we can't assume that
    #   - Kruskal-Wallis uses ranks instead → works with any distribution
    #
    # THE QUESTION: "Is at least one language's sentiment distribution
    # different from the others?"

    if "lang" in df.columns and "sentiment" in df.columns:
        print(f"\n{'─'*40}\nKRUSKAL-WALLIS: Sentiment × Language\n{'─'*40}")

        # Build a dictionary of {language: sentiment_values} for languages
        # with at least 3 sources (can't test with fewer).
        # .values converts from pandas Series to numpy array (needed by scipy)
        groups = {
            name: g["sentiment"].values
            for name, g in df.groupby("lang")  # Split by language
            if len(g) >= 3                       # Keep only groups with 3+ sources
        }

        # Need at least 3 languages with 3+ sources each
        if len(groups) >= 3:
            if HAS_PG:
                # pingouin wants the data in "long format" (one row per observation)
                # with columns for the value and the group label.
                # .isin(groups.keys()) keeps only languages that have 3+ sources
                subset = df[df["lang"].isin(groups.keys())][["sentiment", "lang"]]

                # dv = "dependent variable" (what we're measuring: sentiment)
                # between = "grouping variable" (what we're comparing across: language)
                result = pg.kruskal(data=subset, dv="sentiment", between="lang")
                print(result.to_string())
                report += f"## Kruskal-Wallis\n{result.to_string()}\n\n"
            else:
                # scipy's kruskal takes the groups as separate arguments.
                # *groups.values() "unpacks" the dictionary values:
                # kruskal(group1_scores, group2_scores, group3_scores, ...)
                h, p = sp.kruskal(*groups.values())
                print(f"   H = {h:.2f}, p = {p:.4f}")

    # ════════════════════════════════════════════════════════════════
    # TEST 3: SPEARMAN CORRELATIONS (between numeric variables)
    # ════════════════════════════════════════════════════════════════
    #
    # WHAT IT TESTS: Do two numeric variables move together?
    #   "When sentiment goes up, does something else go up/down too?"
    #
    # WHY SPEARMAN (not Pearson)? Pearson assumes a linear relationship
    # and normal distribution. Spearman uses ranks, so it catches any
    # monotonic relationship (consistently up or consistently down,
    # even if not a straight line).
    #
    # ρ (rho) ranges from -1 to +1:
    #   +1 = perfect positive correlation (one goes up, the other does too)
    #   -1 = perfect negative correlation (one goes up, the other goes down)
    #    0 = no correlation

    if "sentiment" in df.columns:
        print(f"\n{'─'*40}\nSPEARMAN CORRELATIONS\n{'─'*40}")

        # Find all numeric columns in the DataFrame.
        # select_dtypes(include="number") keeps only columns with numeric data.
        # .columns.tolist() converts the column names to a plain list.
        # "idx" (row number) and "bertopic" (topic ID -- a category label,
        # not a quantity) are numeric-looking but meaningless to correlate.
        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("idx", "bertopic")]

        if len(num_cols) >= 2 and HAS_PG:
            # pingouin's pairwise_corr tests ALL pairs of numeric columns at once.
            # method="spearman" specifies Spearman (rank-based) correlation.
            # Returns a DataFrame with: X, Y, r (correlation), p-unc (p-value),
            # power (statistical power), and more.
            corr = pg.pairwise_corr(df[num_cols], method="spearman")

            # Filter to only significant correlations (p < 0.05).
            # "p-unc" = uncorrected p-value. pingouin 0.7 renamed it to
            # "p_unc", so accept either spelling.
            pcol = "p-unc" if "p-unc" in corr.columns else "p_unc"
            sig = corr[corr[pcol] < 0.05]

            if not sig.empty:  # .empty is True if the DataFrame has no rows
                # Show only the most useful columns (that this version has)
                show = [c for c in ["X", "Y", "r", pcol, "power"] if c in sig.columns]
                print(sig[show].to_string(index=False))
                report += f"## Spearman Correlations\n{sig[show].to_string()}\n\n"
            else:
                print("   No significant correlations found")

        elif len(num_cols) >= 2:
            # scipy fallback: test each pair manually.
            # We use a nested loop to avoid testing a pair twice:
            # enumerate gives us the index i, and num_cols[i+1:] gives
            # everything after the current item.
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i+1:]:  # Only pairs we haven't tested yet
                    # .dropna() on BOTH columns together removes rows with a
                    # missing value, so the two series stay paired row-by-row
                    pair = df[[c1, c2]].dropna()
                    rho, p = sp.spearmanr(pair[c1], pair[c2])
                    if p < 0.05:  # Only print significant correlations
                        print(f"   {c1} × {c2}: ρ={rho:.3f}, p={p:.4f}")

    # ════════════════════════════════════════════════════════════════
    # TEST 4: CHI-SQUARED (all categorical pairs)
    # ════════════════════════════════════════════════════════════════
    #
    # This tests EVERY PAIR of categorical variables for association.
    # "Is quality associated with sentiment label?"
    # "Is language associated with topic?"
    # "Is topic associated with sentiment?"
    #
    # combinations(list, 2) generates all unique pairs:
    # combinations(["a","b","c"], 2) → ("a","b"), ("a","c"), ("b","c")

    cat_pairs = []  # Collect categorical columns that exist

    for c in ["quality", "lang", "sent_label"]:
        if c in df.columns:
            cat_pairs.append(c)

    # Add zero-shot topic if it exists
    if "topic_zs" in df.columns:
        cat_pairs.append("topic_zs")

    # Add the BERTopic themes (from analyze) -- sources without usable
    # evidence have no theme, so leave them out of the tests
    if "theme" in df.columns:
        df["theme"] = df["theme"].where(df["theme"] != "no usable evidence")
        cat_pairs.append("theme")

    if len(cat_pairs) >= 2:
        print(f"\n{'─'*40}\nCHI-SQUARED ASSOCIATIONS\n{'─'*40}")

        # combinations generates all unique pairs of categorical columns
        from itertools import combinations

        for c1, c2 in combinations(cat_pairs, 2):
            # Build a contingency table (cross-tab) for this pair
            ct = pd.crosstab(df[c1], df[c2])

            # Need at least 2 categories in each variable
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:

                # ── CHECK CHI-SQUARED ASSUMPTIONS ───────────────────
                # Chi-squared requires that EXPECTED frequencies (not
                # observed) in each cell are ≥ 5. With 80 sources across
                # 6 quality tiers × 3 sentiment labels, many cells will
                # have expected values below 5, making the p-value
                # unreliable.
                #
                # We check by computing the expected frequencies:
                chi2, p, dof, expected = sp.chi2_contingency(ct)

                # Count how many cells have expected frequency < 5
                low_cells = (expected < 5).sum()
                total_cells = expected.size
                pct_low = low_cells / total_cells * 100

                n = ct.sum().sum()
                k = min(ct.shape) - 1
                v = (chi2 / (n * k)) ** 0.5 if k > 0 and n > 0 else 0

                if pct_low > 20:
                    # More than 20% of cells have expected freq < 5.
                    # Chi-squared is unreliable. Try Fisher's exact test.
                    #
                    # Fisher's exact test works on small samples but only
                    # on 2×2 tables in scipy. For larger tables, we report
                    # the chi-squared result WITH a warning.
                    if ct.shape == (2, 2):
                        # 2×2 table → use Fisher's exact test
                        odds, p_fisher = sp.fisher_exact(ct)
                        sig = "*" if p_fisher < 0.05 else " "
                        print(f"   {sig} {c1} × {c2}: Fisher's exact p={p_fisher:.4f}, "
                              f"odds ratio={odds:.2f}")
                        print(f"      (used Fisher's: {pct_low:.0f}% of cells had expected freq < 5)")
                        report += (f"## {c1} × {c2}\nFisher's exact p={p_fisher:.4f}, "
                                   f"odds={odds:.2f}\n\n")
                    else:
                        # Larger table → report chi-squared with warning
                        sig = "*" if p < 0.05 else " "
                        print(f"   {sig} {c1} × {c2}: χ²={chi2:.1f}, p={p:.4f}, V={v:.3f}")
                        print(f"      CAUTION: {pct_low:.0f}% of cells have expected "
                              f"freq < 5 -- p-value may be unreliable")
                        report += (f"## {c1} × {c2}\nχ²={chi2:.1f}, p={p:.4f}, V={v:.3f}\n"
                                   f"{pct_low:.0f}% cells < 5 expected -- interpret with caution\n\n")
                else:
                    # Assumptions met → report normally
                    sig = "*" if p < 0.05 else " "
                    print(f"   {sig} {c1} × {c2}: χ²={chi2:.1f}, p={p:.4f}, V={v:.3f}")
                    report += f"## {c1} × {c2}\nχ²={chi2:.1f}, p={p:.4f}, Cramér's V={v:.3f}\n\n"

    # ════════════════════════════════════════════════════════════════
    # TEST 5: STRUCTURAL EQUATION MODELING (semopy)
    # ════════════════════════════════════════════════════════════════
    #
    # WHAT IS SEM?
    # Chi-squared tells you two variables are ASSOCIATED.
    # SEM tells you the CAUSAL STRUCTURE: which variable affects which,
    # and through what path.
    #
    # OUR MODEL:
    #   source quality → topic discussed → sentiment
    #
    # In plain English: "Vendor sources are more positive. But WHY?
    # Is it because they're vendors (direct effect)? Or because vendor
    # sources tend to discuss 'productivity' (a positive topic) while
    # forums discuss 'job loss' (a negative topic)?"
    #
    # If the indirect path (quality → topic → sentiment) is significant,
    # the topic MEDIATES the relationship. The vendor bias isn't just
    # "vendors are cheerful" -- it's "vendors talk about cheerful topics."
    #
    # WHY SEMOPY AND NOT PINGOUIN'S MEDIATION?
    # pingouin.mediation_analysis uses pd.Categorical().codes, which
    # assigns arbitrary integers to nominal categories (academic=0,
    # vendor=4). That implies "vendor is 4× more academic" -- nonsense.
    #
    # The fix: DUMMY CODING. We create binary 0/1 columns:
    #   quality = "vendor" → is_vendor=1, is_academic=0, is_forum=0
    # Each dummy variable is genuinely numeric (0 or 1), and the
    # relationships are interpretable: "being a vendor source predicts
    # +0.3 on sentiment."
    #
    # semopy is a Python SEM package published in the journal
    # Structural Equation Modeling, uses lavaan-style syntax (the R
    # standard), and outperforms R's lavaan in speed and accuracy.

    if "sentiment" in df.columns and "quality" in df.columns:
        print(f"\n{'─'*40}\nSTRUCTURAL EQUATION MODEL (semopy)\n{'─'*40}")

        try:
            from semopy import Model as SEModel

            # ── Dummy-code the categorical variables ────────────────
            # pd.get_dummies converts a categorical column into binary columns:
            #   quality = ["academic", "vendor", "forum"]
            #   becomes: quality_academic (0/1), quality_vendor (0/1), quality_forum (0/1)
            #
            # drop_first=True removes one column to avoid multicollinearity
            # (if it's not academic AND not vendor AND not forum... it must be
            # the dropped category. Including all of them would be redundant
            # and break the math.)
            quality_dummies = pd.get_dummies(df["quality"], prefix="q", drop_first=True).astype(int)
            # .astype(int): pandas 2.x returns True/False columns by default;
            # semopy needs numbers, so we convert to 0/1 integers
            sem_df = pd.concat([df[["sentiment"]], quality_dummies], axis=1)

            # Add topic dummies if available
            has_topic = "topic_zs" in df.columns
            if has_topic:
                topic_dummies = pd.get_dummies(df["topic_zs"], prefix="t", drop_first=True).astype(int)
                sem_df = pd.concat([sem_df, topic_dummies], axis=1)

            # Clean column names (semopy doesn't like spaces or special chars)
            # (No other renaming: the q_/t_ prefixes must survive, because
            # the q_cols / t_cols lookups below depend on them.)
            sem_df.columns = [c.replace(" ", "_").replace("/", "_").replace(".", "_")
                              .replace("-", "_")
                              for c in sem_df.columns]

            # Drop rows with any missing values (SEM needs complete cases)
            sem_df = sem_df.dropna()

            # ── Build the SEM model specification ───────────────────
            # semopy uses lavaan-style syntax:
            #   y ~ x1 + x2    means "y is predicted by x1 and x2"
            #
            # We build two models:
            #   A) Direct only: sentiment ~ quality dummies
            #   B) Mediated: sentiment ~ quality dummies + topic dummies
            #                 topic dummies ~ quality dummies
            #
            # If adding topics reduces the quality→sentiment path,
            # that's evidence of mediation.

            # Which dummies go into the model (we use at most 4 of each to
            # avoid overfitting). Skip categories with fewer than 3 sources --
            # a path estimated from 1 source is noise -- then take the biggest,
            # always keeping vendor first since vendor bias is the hypothesis.
            # (Taking the first 4 alphabetically used to pick n=1 tiers like
            # "consulting" and leave vendor out entirely.)
            def pick(prefix):
                cols = [c for c in sem_df.columns
                        if c.startswith(prefix) and sem_df[c].sum() >= 3]
                return sorted(cols, key=lambda c: ("vendor" not in c, -sem_df[c].sum()))

            q_cols = pick("q_")
            t_cols = pick("t_")

            if q_cols and len(sem_df) >= 30:
                q_predictors = " + ".join(q_cols[:4])  # Limit to 4 to avoid overfit

                if has_topic and t_cols:
                    t_predictors = " + ".join(t_cols[:4])
                    # Full mediated model:
                    # sentiment depends on quality AND topic
                    # topic depends on quality (the mediation path)
                    model_spec = (
                        f"sentiment ~ {q_predictors} + {t_predictors}\n"
                        + "\n".join(f"{tc} ~ {q_predictors}" for tc in t_cols[:4])
                    )
                    model_label = "mediated (quality → topic → sentiment)"
                else:
                    # Direct-only model (no topic data)
                    model_spec = f"sentiment ~ {q_predictors}"
                    model_label = "direct (quality → sentiment)"

                print(f"   Model: {model_label}")
                print(f"   Observations: {len(sem_df)}")

                # Fit the model
                sem_model = SEModel(model_spec)
                sem_model.fit(sem_df)

                # Get the results table
                results = sem_model.inspect()
                # Filter to the interesting paths (not variances)
                paths = results[results["op"] == "~"]

                if not paths.empty:
                    print("\n   Path coefficients:")
                    for _, row in paths.iterrows():
                        lval = row.get("lval", "")
                        rval = row.get("rval", "")
                        est = row.get("Estimate", 0)
                        pval = row.get("p-value", 1)
                        sig = "*" if pval < 0.05 else " "
                        print(f"   {sig} {rval:30s} → {lval:15s}  β={est:+.3f}  p={pval:.4f}")

                    report += f"## SEM: {model_label}\n{paths.to_string()}\n\n"
                else:
                    print("   No significant paths found")
            elif not q_cols:
                print("   Need 2+ source quality tiers for SEM")
            else:
                print(f"   Need 30+ observations for SEM (have {len(sem_df)})")

        except ImportError:
            print("   pip install semopy  (for structural equation modeling)")
            print("   Falling back to chi-squared associations (already reported above)")

        except Exception as e:
            print(f"   SEM error: {e}")
            print("   This can happen with very small samples or too many categories.")

    # ════════════════════════════════════════════════════════════════
    # BONUS: RESEARCHPY SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════
    #
    # researchpy formats statistical summaries in APA style,
    # ready to paste into a paper. It's designed for researchers
    # who need publication-ready output.

    try:
        import researchpy as rp

        print(f"\n{'─'*40}\nRESEARCHPY SUMMARY TABLE\n{'─'*40}")

        if "sentiment" in df.columns and "quality" in df.columns:
            # summary_cont gives: N, mean, std, SE, and 95% CI
            # for each group (quality tier)
            summary = rp.summary_cont(df.groupby("quality")["sentiment"])
            print(summary.to_string())

    except ImportError:
        pass  # researchpy not installed -- skip silently
    except Exception as e:
        # researchpy 0.3.7.3 fails to IMPORT on Python 3.10+ (a TypeError in
        # its own type hints), and its summary can break on newer pandas.
        # It's a bonus table -- never let it crash the stats run.
        print(f"   researchpy unavailable ({type(e).__name__}) -- skipped")

    # ── Save everything to scratchpad ───────────────────────────────
    pad("stats_report", report)
    print(f"\n{'='*60}\nSTATS COMPLETE\n{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# PATH 2: SURVEY STATS (analyzing external survey CSV data)
# ════════════════════════════════════════════════════════════════════════

def _survey_stats(df, args):
    """
    Run statistical tests on an external survey dataset (e.g., Stack
    Overflow Developer Survey).

    This function auto-detects what's in the data and runs appropriate
    tests. It looks for AI-related columns and tries logistic regression
    on any binary (yes/no) AI variable it finds.
    """

    import pandas as pd

    # Show basic info about the dataset
    # .shape returns (rows, columns) -- like the size of a spreadsheet
    print(f"   Shape: {df.shape}")
    # Show first 15 column names so the user knows what's in the data
    print(f"   Columns: {list(df.columns)[:15]}")

    # ── Auto-detect AI-related columns ──────────────────────────────
    # Search column names for keywords related to our topic.
    # If a column is called "uses_ai_tools" or "copilot_adoption",
    # we want to find it automatically.
    ai_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["ai", "copilot", "assistant", "tool", "llm"])
    ]

    if ai_cols:
        print(f"\n   AI columns: {ai_cols}")

    report = f"# Survey Analysis\n\nRows: {len(df)}\n\n"

    # ── Logistic regression on binary AI variables ──────────────────
    #
    # WHAT IS LOGISTIC REGRESSION?
    # Regular (linear) regression predicts a NUMBER: "how much revenue?"
    # Logistic regression predicts a YES/NO outcome: "does this developer
    # use AI tools?"
    #
    # It tells you which factors PREDICT the outcome. For example:
    # "Controlling for everything else, developers with 5+ years of
    # experience are 2.3x more likely to use AI coding tools."
    #
    # We auto-detect binary (yes/no) AI-related columns and run
    # logistic regression with all available numeric columns as predictors.
    #
    # .nunique() returns how many unique values a column has.
    # If it's 2 (e.g., "Yes"/"No"), it's binary → good for logistic regression.

    for col in ai_cols:
        if df[col].nunique() == 2:  # Binary column (yes/no, true/false, 0/1)
            print(f"\n{'─'*40}\nLOGISTIC REGRESSION: {col}\n{'─'*40}")

            try:
                import statsmodels.api as sm

                # Find all numeric columns to use as predictors.
                # select_dtypes(include="number") keeps only numeric columns.
                nums = df.select_dtypes(include="number").columns.tolist()

                # Use up to 5 numeric columns as predictors (exclude the target itself).
                # More predictors need more data to be reliable.
                preds = [c for c in nums if c != col][:5]
                if not preds:
                    print("   No numeric predictor columns -- skipping")
                    continue

                # Drop rows with a missing target OR predictor in one go
                # (logistic regression crashes on NaN values, and a missing
                # target would otherwise become code -1 below).
                data = df[[col] + preds].dropna()

                # Convert the yes/no column to 0/1 numbers.
                # pd.Categorical(...).codes assigns 0 to one value and 1 to the other.
                y = pd.Categorical(data[col]).codes
                X = data[preds]

                # sm.add_constant adds a column of 1's (the "intercept" term).
                # Without it, the model is forced through the origin, which
                # is almost never what you want.
                X = sm.add_constant(X)

                # Fit the logistic regression model.
                # sm.Logit(y, X) creates the model.
                # .fit(disp=0) runs it (disp=0 = don't print fitting progress).
                model = sm.Logit(y, X).fit(disp=0)

                # .summary2().tables[1] gives the coefficient table:
                # for each predictor, it shows the coefficient (effect size),
                # standard error, z-score, p-value, and 95% CI.
                print(model.summary2().tables[1].to_string())
                report += f"## Logistic: {col}\n{model.summary2().tables[1].to_string()}\n\n"

            except Exception as e:
                print(f"   Error: {e}")

    # Save to scratchpad
    pad("stats_survey", report)
    print(f"\n{'='*60}\nSURVEY STATS COMPLETE\n{'='*60}")
