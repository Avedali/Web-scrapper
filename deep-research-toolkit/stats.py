"""
stats.py: Mode 2b, statistical tests on analyze mode's output or a survey CSV.
Run: python agent.py stats [--data survey.csv] (or launcher.py, option 3)
Corpus (default): Mann-Whitney, Kruskal-Wallis, Spearman, chi-squared, SEM on corpus_analyzed.csv
Survey (--data): logistic regression on each binary AI-related column
Uses pingouin if installed (effect sizes, power, Bayes Factor), else scipy.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json
import sys
import os

from core import pad


# ════════════════════════════════════════════════════════════════════════
# MAIN: corpus or survey path
# ════════════════════════════════════════════════════════════════════════


MIN_GROUP = 10   # Categories with fewer sources are merged into "other" for chi-squared

def run_stats(args):
    """Run survey stats if --data is given, else corpus stats."""

    print("=" * 60)
    print("MODE 2b: STATISTICAL ANALYSIS")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    if args.data and os.path.exists(args.data):
        print(f"   External data: {args.data}")
        df = pd.read_csv(args.data)
        _survey_stats(df, args)
    else:
        csv_path = os.path.join(args.corpus, "corpus_analyzed.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            _corpus_stats(df, args)
        else:
            print("Run 'python agent.py analyze' first, or pass --data file.csv")


# ════════════════════════════════════════════════════════════════════════
# PATH 1: CORPUS STATS
# ════════════════════════════════════════════════════════════════════════

def _corpus_stats(df, args):
    """Run the corpus tests on analyze mode's columns; saves scratchpad stats_report and stats_results.json."""
    import pandas as pd
    results = {}   # Key test results, for the charts (viz)

    report = "# Statistical Analysis\n\n"

    try:
        import pingouin as pg
        HAS_PG = True
        print("   Using pingouin (effect sizes + Bayes Factor included)")
    except ImportError:
        HAS_PG = False
        print("   pip install pingouin (falling back to scipy)")

    try:
        from scipy import stats as sp
    except ImportError:
        sys.exit("pip install scipy")

    # ── TEST 0: Descriptive statistics ──────────────────────────────
    if "sentiment" in df.columns:
        print(f"\n{'─'*40}\nDESCRIPTIVE\n{'─'*40}")

        n = len(df)
        m = df["sentiment"].mean()
        se = df["sentiment"].std() / (n ** 0.5)

        print(f"   Mean sentiment: {m:+.3f} (95% CI: [{m-1.96*se:.3f}, {m+1.96*se:.3f}])")
        print(f"   Std: {df['sentiment'].std():.3f}, n={n}")

        report += (f"## Descriptive\n"
                   f"Mean: {m:+.3f}, 95% CI: [{m-1.96*se:.3f}, {m+1.96*se:.3f}], n={n}\n\n")

    # ── TEST 1: Mann-Whitney U (vendor vs non-vendor) ───────────────
    # Checks whether vendor and other sources differ in sentiment (rank-based)
    if "quality" in df.columns and "sentiment" in df.columns:
        print(f"\n{'─'*40}\nMANN-WHITNEY U: Vendor vs Non-Vendor\n{'─'*40}")

        vendor = df[df["quality"].str.contains("vendor", case=False, na=False)]["sentiment"]

        non_vendor = df[~df["quality"].str.contains("vendor", case=False, na=False)]["sentiment"]

        if len(vendor) >= 3 and len(non_vendor) >= 3:
            if HAS_PG:
                result = pg.mwu(vendor, non_vendor)
                print(result.to_string())
                # Column is "p-val" or "p_val" depending on pingouin version
                p = float(result[[c for c in result.columns if c.replace("-", "_") == "p_val"][0]].iloc[0])
                report += (f"## Mann-Whitney U\n{result.to_string()}\n"
                           + _mwu_in_words(vendor, non_vendor, p) + "\n")
            else:
                u, p = sp.mannwhitneyu(vendor, non_vendor, alternative="two-sided")
                print(f"   U = {u:.1f}, p = {p:.4f}")
                print(f"   Vendor: {vendor.mean():+.3f} (n={len(vendor)})")
                print(f"   Non-vendor: {non_vendor.mean():+.3f} (n={len(non_vendor)})")
                if p < 0.05:
                    print("   Significant")
                else:
                    print("   Not significant")
                report += (f"## Mann-Whitney U\nU={u:.1f}, p={p:.4f}\n"
                           + _mwu_in_words(vendor, non_vendor, p) + "\n")

            results["vendor_vs_other"] = {"p": p, "n_vendor": len(vendor), "n_other": len(non_vendor),
                                          "mean_vendor": float(vendor.mean()), "mean_other": float(non_vendor.mean())}

    # ── TEST 2: Kruskal-Wallis (sentiment across languages) ─────────
    # Checks whether sentiment differs across 3+ languages with 3+ sources each
    if "lang" in df.columns and "sentiment" in df.columns:
        print(f"\n{'─'*40}\nKRUSKAL-WALLIS: Sentiment × Language\n{'─'*40}")

        groups = {
            name: g["sentiment"].values
            for name, g in df.groupby("lang")
            if len(g) >= 3
        }

        if len(groups) >= 3:
            if HAS_PG:
                subset = df[df["lang"].isin(groups.keys())][["sentiment", "lang"]]

                result = pg.kruskal(data=subset, dv="sentiment", between="lang")
                print(result.to_string())
                p = float(result[[c for c in result.columns if c.replace("-", "_") == "p_unc"][0]].iloc[0])
                report += (f"## Kruskal-Wallis\n{result.to_string()}\n"
                           + _kruskal_in_words(groups, p) + "\n")
            else:
                h, p = sp.kruskal(*groups.values())
                print(f"   H = {h:.2f}, p = {p:.4f}")
                report += (f"## Kruskal-Wallis\nH={h:.2f}, p={p:.4f}\n"
                           + _kruskal_in_words(groups, p) + "\n")

            results["languages"] = {"p": p, "languages": sorted(groups)}

    # ── TEST 3: Spearman correlations ───────────────────────────────
    # Checks whether numeric columns rise or fall together (rank-based)
    if "sentiment" in df.columns:
        print(f"\n{'─'*40}\nSPEARMAN CORRELATIONS\n{'─'*40}")

        # idx and bertopic are labels, not quantities
        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("idx", "bertopic")]

        if len(num_cols) >= 2 and HAS_PG:
            corr = pg.pairwise_corr(df[num_cols], method="spearman")

            pcol = "p-unc" if "p-unc" in corr.columns else "p_unc"
            sig = corr[corr[pcol] < 0.05]

            if not sig.empty:
                show = [c for c in ["X", "Y", "r", pcol, "power"] if c in sig.columns]
                print(sig[show].to_string(index=False))
                report += f"## Spearman Correlations\n{sig[show].to_string()}\n\n"
            else:
                print("   No significant correlations found")

        elif len(num_cols) >= 2:
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i+1:]:
                    pair = df[[c1, c2]].dropna()
                    rho, p = sp.spearmanr(pair[c1], pair[c2])
                    if p < 0.05:
                        print(f"   {c1} × {c2}: ρ={rho:.3f}, p={p:.4f}")

    # ── TEST 4: Chi-squared (every categorical pair) ────────────────
    # Checks whether two categorical columns are associated
    cat_pairs = []

    for c in ["quality", "lang", "sent_label"]:
        if c in df.columns:
            cat_pairs.append(c)

    if "topic_zs" in df.columns:
        cat_pairs.append("topic_zs")

    if "theme" in df.columns:
        df["theme"] = df["theme"].where(df["theme"] != "no usable evidence")
        cat_pairs.append("theme")

    if len(cat_pairs) >= 2:
        print(f"\n{'─'*40}\nCHI-SQUARED ASSOCIATIONS\n{'─'*40}")

        from itertools import combinations

        # Categories with fewer than MIN_GROUP sources are merged into "other",
        # so each cell can reach the 5 expected cases the test needs
        merged = df[cat_pairs].copy()
        for c in cat_pairs:
            counts = merged[c].value_counts()
            small = counts[counts < MIN_GROUP].index
            merged[c] = merged[c].where(~merged[c].isin(small), "other")

        for c1, c2 in combinations(cat_pairs, 2):
            ct = pd.crosstab(merged[c1], merged[c2])

            if ct.shape[0] >= 2 and ct.shape[1] >= 2:

                chi2, p, dof, expected = sp.chi2_contingency(ct)

                low_cells = (expected < 5).sum()
                total_cells = expected.size
                pct_low = low_cells / total_cells * 100

                n = ct.sum().sum()
                k = min(ct.shape) - 1
                v = (chi2 / (n * k)) ** 0.5 if k > 0 and n > 0 else 0

                # Over 20% of cells expected < 5: Fisher's exact on 2×2, else a warning
                if pct_low > 20:
                    if ct.shape == (2, 2):
                        odds, p_fisher = sp.fisher_exact(ct)
                        sig = "*" if p_fisher < 0.05 else " "
                        print(f"   {sig} {c1} × {c2}: Fisher's exact p={p_fisher:.4f}, "
                              f"odds ratio={odds:.2f}")
                        print(f"      (used Fisher's: {pct_low:.0f}% of cells had expected freq < 5)")
                        report += (f"## {c1} × {c2}\nFisher's exact p={p_fisher:.4f}, "
                                   f"odds={odds:.2f}\n\n")
                    else:
                        # Still too sparse after merging: a p-value here would not be valid
                        print(f"     {c1} × {c2}: not tested ({pct_low:.0f}% of cells expect fewer "
                              "than 5 sources; the test needs at most 20%)")
                        report += (f"## {c1} × {c2}\nNot tested: {pct_low:.0f}% of cells expect fewer "
                                   "than 5 sources, even after merging small groups (the test needs at "
                                   "most 20%). More sources per group are needed.\n\n")
                else:
                    sig = "*" if p < 0.05 else " "
                    print(f"   {sig} {c1} × {c2}: χ²={chi2:.1f}, p={p:.4f}, V={v:.3f}")
                    report += f"## {c1} × {c2}\nχ²={chi2:.1f}, p={p:.4f}, Cramér's V={v:.3f}\n\n"

    # ── TEST 5: Structural equation model (semopy) ──────────────────
    # Checks whether topic mediates quality → sentiment; dummy-coded, not integer codes
    if "sentiment" in df.columns and "quality" in df.columns:
        print(f"\n{'─'*40}\nSTRUCTURAL EQUATION MODEL (semopy)\n{'─'*40}")

        try:
            from semopy import Model as SEModel

            quality_dummies = pd.get_dummies(df["quality"], prefix="q", drop_first=True).astype(int)
            sem_df = pd.concat([df[["sentiment"]], quality_dummies], axis=1)

            has_topic = "topic_zs" in df.columns
            if has_topic:
                topic_dummies = pd.get_dummies(df["topic_zs"], prefix="t", drop_first=True).astype(int)
                sem_df = pd.concat([sem_df, topic_dummies], axis=1)

            sem_df.columns = [c.replace(" ", "_").replace("/", "_").replace(".", "_")
                              .replace("-", "_")
                              for c in sem_df.columns]

            sem_df = sem_df.dropna()

            # Dummies with 3+ sources, vendor first, then largest
            def pick(prefix):
                cols = [c for c in sem_df.columns
                        if c.startswith(prefix) and sem_df[c].sum() >= 3]
                return sorted(cols, key=lambda c: ("vendor" not in c, -sem_df[c].sum()))

            q_cols = pick("q_")
            t_cols = pick("t_")

            if q_cols and len(sem_df) >= 30:
                q_predictors = " + ".join(q_cols[:4])

                if has_topic and t_cols:
                    t_predictors = " + ".join(t_cols[:4])
                    model_spec = (
                        f"sentiment ~ {q_predictors} + {t_predictors}\n"
                        + "\n".join(f"{tc} ~ {q_predictors}" for tc in t_cols[:4])
                    )
                    model_label = "mediated (quality → topic → sentiment)"
                else:
                    model_spec = f"sentiment ~ {q_predictors}"
                    model_label = "direct (quality → sentiment)"

                print(f"   Model: {model_label}")
                print(f"   Observations: {len(sem_df)}")

                sem_model = SEModel(model_spec)
                sem_model.fit(sem_df)

                results = sem_model.inspect()
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

    # ── Optional: researchpy APA summary per quality tier ───────────
    try:
        import researchpy as rp

        print(f"\n{'─'*40}\nRESEARCHPY SUMMARY TABLE\n{'─'*40}")

        if "sentiment" in df.columns and "quality" in df.columns:
            summary = rp.summary_cont(df.groupby("quality")["sentiment"])
            print(summary.to_string())

    except ImportError:
        pass
    except Exception as e:
        # Fails to import on Python 3.10+ and on newer pandas; optional table
        print(f"   researchpy unavailable ({type(e).__name__}) -- skipped")

    # ── Save ────────────────────────────────────────────────────────
    pad("stats_report", report)
    with open(os.path.join(args.corpus, "stats_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n{'='*60}\nSTATS COMPLETE\n{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# PATH 2: SURVEY STATS
# ════════════════════════════════════════════════════════════════════════

def _survey_stats(df, args):
    """Run logistic regression on each binary AI-related column of a survey CSV."""

    import pandas as pd

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:15]}")

    ai_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["ai", "copilot", "assistant", "tool", "llm"])
    ]

    if ai_cols:
        print(f"\n   AI columns: {ai_cols}")

    report = f"# Survey Analysis\n\nRows: {len(df)}\n\n"

    # ── Logistic regression: which numeric columns predict a yes/no AI column ─
    for col in ai_cols:
        if df[col].nunique() == 2:
            print(f"\n{'─'*40}\nLOGISTIC REGRESSION: {col}\n{'─'*40}")

            try:
                import statsmodels.api as sm

                nums = df.select_dtypes(include="number").columns.tolist()

                preds = [c for c in nums if c != col][:5]
                if not preds:
                    print("   No numeric predictor columns -- skipping")
                    continue

                # Drop NaN rows first, or a missing target becomes code -1
                data = df[[col] + preds].dropna()

                y = pd.Categorical(data[col]).codes
                X = data[preds]

                X = sm.add_constant(X)

                model = sm.Logit(y, X).fit(disp=0)

                print(model.summary2().tables[1].to_string())
                report += f"## Logistic: {col}\n{model.summary2().tables[1].to_string()}\n\n"

            except Exception as e:
                print(f"   Error: {e}")

    pad("stats_survey", report)
    print(f"\n{'='*60}\nSURVEY STATS COMPLETE\n{'='*60}")


# ════════════════════════════════════════════════════════════════════════
# PLAIN-WORDS SUMMARIES: what each test compared and found
# ════════════════════════════════════════════════════════════════════════
# Stops the brief's writer from reading a sentiment test as proof AI tools work

def _p(p):
    """Format a p-value: 'p=0.0037', or 'p<0.0001' instead of 'p=0.0000'."""
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def _mwu_in_words(vendor, non_vendor, p):
    """One sentence saying what the Mann-Whitney test compared and found."""
    if p >= 0.05:
        finding = "No significant difference in sentiment between them"
    elif vendor.mean() > non_vendor.mean():
        finding = "Vendor/primary sources are significantly MORE positive than the rest"
    else:
        finding = "Vendor/primary sources are significantly LESS positive than the rest"
    return (f"In words: compares the sentiment of sources classified as vendor/primary "
            f"(n={len(vendor)}, mean {vendor.mean():+.3f}) with all other sources "
            f"(n={len(non_vendor)}, mean {non_vendor.mean():+.3f}). {finding} ({_p(p)}). "
            f"This is about how sources talk, not about whether AI tools work.\n")


def _kruskal_in_words(groups, p):
    """One sentence saying what the Kruskal-Wallis test compared and found."""
    means = sorted(((sum(v) / len(v), lang) for lang, v in groups.items()), reverse=True)
    finding = ("Sentiment differs significantly between languages" if p < 0.05
               else "No significant difference in sentiment between languages")
    return (f"In words: compares the sentiment of sources across {len(groups)} languages. "
            f"{finding} ({_p(p)}). Most positive: {means[0][1]} ({means[0][0]:+.2f}); "
            f"least positive: {means[-1][1]} ({means[-1][0]:+.2f}).\n")
