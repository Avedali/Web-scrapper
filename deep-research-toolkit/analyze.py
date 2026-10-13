"""analyze.py: Mode 2a, NLP analysis of the explore corpus with Hugging Face models.
Run: python agent.py analyze (or launcher.py, option 2)
How it works:
  Sentiment: scores each source positive, negative or neutral (multilingual BERT)
  Zero-shot topics: sorts each source into fixed research categories (XNLI)
  NER: counts the most mentioned organizations
  Themes: clusters evidence cards with BERTopic and names each theme by AI
  Chi-squared: tests whether sentiment depends on source type
Saves corpus_analyzed.csv, analysis_results.json and the analyze_report pad.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json
import sys
import os
import re

from collections import defaultdict

from core import (
    pad,
    call_ai,
    build_evidence_cards,
    NO_EVIDENCE,
)

ANALYZE_CHARS = None  # Characters of each source the models read (None = all)

# Apple Silicon: run GPU-unsupported ops on the CPU (set before torch loads)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class _FallbackPipe:
    """A model on the Mac GPU that switches to the CPU, and stays there, if the GPU fails."""

    def __init__(self, pipe, make_cpu_pipe):
        self.pipe, self.make_cpu_pipe, self.on_gpu = pipe, make_cpu_pipe, True

    def __call__(self, *args, **kwargs):
        try:
            return self.pipe(*args, **kwargs)
        except Exception as e:
            if not self.on_gpu:
                raise
            print(f"      Mac GPU failed ({type(e).__name__}) -- switching to the CPU", flush=True)
            self.pipe, self.on_gpu = self.make_cpu_pipe(), False
            return self.pipe(*args, **kwargs)


def _make_pipeline(task, test, **kwargs):
    """Build a HF pipeline on the Mac GPU if `test(pipe)` passes there, else on the CPU/default device."""
    from transformers import pipeline
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            pipe = pipeline(task, device="mps", **kwargs)
            test(pipe)
            print("   (running on the Mac's GPU)")
            return _FallbackPipe(pipe, lambda: pipeline(task, device="cpu", **kwargs))
        except Exception as e:
            print(f"   Mac GPU not usable here ({type(e).__name__}) -- using the CPU")
            return pipeline(task, device="cpu", **kwargs)
    return pipeline(task, **kwargs)


# ════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING
# ════════════════════════════════════════════════════════════════════════

def _progress(label, total):
    """Return a tick() that prints progress and time left every ~10% of documents."""
    import time
    state = {"done": 0, "start": time.time()}
    step = max(1, total // 10)

    def tick():
        state["done"] += 1
        done = state["done"]
        if done % step == 0 or done == total:
            spent = time.time() - state["start"]
            left = spent / done * (total - done)
            print(f"      {label}: {done}/{total} done, "
                  f"~{left / 60:.0f} min left", flush=True)
    return tick


def _chunk_text(text, chunk_size=450, overlap=50):
    """Split text into overlapping chunks of chunk_size characters (fits the 512-token model limit)."""
    if not text or len(text) <= chunk_size:
        return [text] if text else [""]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_analyze(args):
    """Run all NLP analyses on the corpus from explore mode."""

    # ── Load the corpus ─────────────────────────────────────────────
    corpus_path = os.path.join(args.corpus, "corpus.json")

    if not os.path.exists(corpus_path):
        sys.exit("No corpus. Run 'python agent.py explore' first.")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    if not corpus:
        sys.exit("Corpus is empty. Re-run 'python agent.py explore'.")

    print("=" * 60)
    print(f"MODE 2a: ANALYZE ({len(corpus)} documents)")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    df = pd.DataFrame(corpus)

    df["content"] = df["content"].fillna("")
    if ANALYZE_CHARS:
        df["content"] = df["content"].str.slice(0, ANALYZE_CHARS)

    # ── Evidence cards: cached English summary per source ──────────
    api_key = getattr(args, "key", None) or os.environ.get("OPENROUTER_API_KEY")
    model = getattr(args, "model", None) or "google/gemini-2.5-flash-lite"
    cards = build_evidence_cards(corpus, api_key, model, args.corpus) if api_key else {}
    if not api_key:
        print("   No API key -- topics and themes will use the raw pages "
              "(set a key for English evidence cards)")

    def card_or_text(row):
        """The source's evidence card if it has a usable one, else its text."""
        card = cards.get(row["idx"], "")
        return card if card and not card.startswith(NO_EVIDENCE) else row["content"]

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 1: SENTIMENT
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'─'*40}\nSENTIMENT (Hugging Face -- multilingual)\n{'─'*40}")

    try:
        sentiment_pipe = _make_pipeline(
            "sentiment-analysis",
            test=lambda pipe: pipe("This tool saves me time."),
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512,
        )

        def get_sentiment(text):
            """Return (score -1..+1, label): 1-5 stars averaged over all chunks."""
            try:
                chunks = _chunk_text(text)
                scores = []

                for chunk in chunks:
                    r = sentiment_pipe(chunk)[0]
                    stars = int(r["label"][0])
                    scores.append((stars - 3) / 2.0)

                avg_score = sum(scores) / len(scores) if scores else 0.0

                if avg_score > 0.15:
                    label = "POSITIVE"
                elif avg_score < -0.15:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"

                return avg_score, label
            except Exception:
                return 0.0, "NEUTRAL"

        tick = _progress("sentiment", len(df))

        def sentiment_with_progress(text):
            result = get_sentiment(text)
            tick()
            return result

        sentiments = df["content"].apply(sentiment_with_progress)

        df["sentiment"] = sentiments.apply(lambda x: x[0])
        df["sent_label"] = sentiments.apply(lambda x: x[1])

        sent_by_quality = df.groupby("quality")["sentiment"].agg(["mean", "count"])
        print("\n   By source type:")
        for q, row in sent_by_quality.iterrows():
            print(f"      {q:30s}  {row['mean']:+.3f} (n={int(row['count'])})")

        sent_by_lang = df.groupby("lang")["sentiment"].agg(["mean", "count"])
        print("\n   By language:")
        for lang, row in sent_by_lang.iterrows():
            print(f"      {lang:5s}  {row['mean']:+.3f} (n={int(row['count'])})")

    except Exception as e:
        # Missing install or failed model download: fall back to TextBlob
        print(f"   HF sentiment unavailable ({type(e).__name__}: {str(e)[:150]})")
        print("   Falling back to TextBlob...")
        try:
            from textblob import TextBlob
            df["sentiment"] = df["content"].apply(lambda t: TextBlob(t).sentiment.polarity)
            df["sent_label"] = df["sentiment"].apply(
                lambda s: "POSITIVE" if s > 0.05 else ("NEGATIVE" if s < -0.05 else "NEUTRAL")
            )
        except ImportError:
            print("   No sentiment library available")
            df["sentiment"] = 0.0
            df["sent_label"] = "UNKNOWN"

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 2: ZERO-SHOT CLASSIFICATION
    # ════════════════════════════════════════════════════════════════

    print("\n" + '─'*40 + "\n ZERO-SHOT CLASSIFICATION (HF -- multilingual)\n" + '─'*40)

    try:
        zs_pipe = _make_pipeline(
            "zero-shot-classification",
            test=lambda pipe: pipe("This tool saves me time.", candidate_labels=["time", "cost"]),
            model="joeddav/xlm-roberta-large-xnli",
            truncation=True,
        )

        categories = [
            "developer productivity and efficiency",
            "workforce displacement and job loss concerns",
            "code quality and security risks",
            "industry adoption and market trends",
            "regulation and ethics",
            "developer experience and satisfaction",
        ]

        def classify(text):
            """Return the category that wins the most chunks of text."""
            try:
                chunks = _chunk_text(text)
                votes = defaultdict(int)

                for chunk in chunks:
                    r = zs_pipe(chunk, candidate_labels=categories)
                    winner = r["labels"][0]
                    votes[winner] += 1

                return max(votes, key=votes.get) if votes else "unclassified"
            except Exception:
                return "unclassified"

        print(f"   Classifying {len(df)} documents into {len(categories)} categories...")

        # Classify the evidence card where there is one (about 7x faster than raw text)
        tick = _progress("zero-shot", len(df))

        def classify_with_progress(text):
            result = classify(text)
            tick()
            return result

        df["topic_zs"] = df.apply(card_or_text, axis=1).apply(classify_with_progress)

        topic_counts = df["topic_zs"].value_counts()
        print("\n   Topic distribution:")
        for topic, count in topic_counts.items():
            print(f"      {topic:45s}  {count}")

        ct = pd.crosstab(df["lang"], df["topic_zs"])

        if ct.shape[0] >= 2:
            print("\n   Topic × Language cross-tab:")
            print(ct.to_string())

    except Exception as e:
        print(f"   zero-shot unavailable ({type(e).__name__}: {str(e)[:150]})")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 3: NAMED ENTITY RECOGNITION
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'─'*40}\nNAMED ENTITY RECOGNITION (HF -- multilingual)\n{'─'*40}")

    try:
        ner_pipe = _make_pipeline(
            "ner",
            test=lambda pipe: pipe("GitHub and Google work with developers in Paris."),
            aggregation_strategy="simple",
            model="Davlan/xlm-roberta-large-ner-hrl",
            # No truncation=True: the NER pipeline rejects it
        )

        all_entities = defaultdict(int)

        tick = _progress("NER", len(df))
        for _, row in df.iterrows():
            tick()
            try:
                chunks = _chunk_text(row["content"])

                for chunk in chunks:
                    entities = ner_pipe(chunk)

                    for e in entities:
                        # Keep confident ORG names, skip empty words and bare punctuation
                        word = e["word"].strip()
                        is_name = any(ch.isalnum() for ch in word)
                        if e["entity_group"] == "ORG" and e["score"] > 0.8 and is_name:
                            all_entities[word] += 1
            except Exception:
                pass

        top_orgs = sorted(all_entities.items(), key=lambda x: -x[1])[:20]

        print("   Most mentioned organizations:")
        for org, count in top_orgs:
            print(f"      {org:30s}  {count} mentions")

    except Exception as e:
        print(f"   NER unavailable ({type(e).__name__}: {str(e)[:150]})")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 4: THEMES (BERTopic on evidence cards)
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'─'*40}\nTHEMES (BERTopic)\n{'─'*40}")

    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer

        # ── Cluster evidence cards if present, else raw pages ──────
        if cards:
            keep = [i for i, idx in enumerate(df["idx"])
                    if idx in cards and not cards[idx].startswith(NO_EVIDENCE)]
            docs = [cards[df["idx"].iloc[i]] for i in keep]
        else:
            keep = list(range(len(df)))
            docs = df["content"].tolist()

        if len(docs) >= 10:
            n_themes = max(5, min(10, len(docs) // 12))

            # K-means: HDBSCAN put ~90% of these same-subject cards in one topic
            topic_model = BERTopic(
                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                hdbscan_model=KMeans(n_clusters=n_themes, random_state=42, n_init=10),
                vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                representation_model=KeyBERTInspired(),
                verbose=False,
            )

            topics, _ = topic_model.fit_transform(docs)

            # ── Name each theme in English (one AI call) ────────────
            names = {}
            if cards:
                by_topic = {}
                for t, d in zip(topics, docs, strict=True):
                    by_topic.setdefault(t, []).append(d)
                listing = "\n\n".join(
                    f"THEME {t}:\n" + "\n".join(f"- {d[:200]}" for d in ds[:6])
                    for t, ds in sorted(by_topic.items()))
                reply = call_ai(
                    "Each THEME below is a cluster of research summaries. Give each "
                    "a short, specific English name (3-7 words) for what its sources "
                    "have in common. One line per theme: 'THEME <n>: <name>'.\n\n" + listing,
                    api_key, model, temp=0.2)
                names = {int(n): name.strip(" *") for n, name in
                         re.findall(r"THEME\s*(\d+)\s*[:：]\s*(.+)", reply)}
            for t in set(topics):
                if t not in names:
                    names[t] = ", ".join(w for w, _ in topic_model.get_topic(t)[:4])
            topic_model.set_topic_labels(names)

            theme_ids = [-1] * len(df)
            for i, t in zip(keep, topics, strict=True):
                theme_ids[i] = int(t)
            df["bertopic"] = theme_ids
            df["theme"] = [names.get(t, "no usable evidence") for t in theme_ids]

            print(f"   {n_themes} themes from {len(docs)} "
                  f"{'evidence cards' if cards else 'pages'}:")
            for t, count in df.loc[df["bertopic"] != -1, "bertopic"].value_counts().items():
                langs = df.loc[df["bertopic"] == t, "lang"].value_counts()
                lang_mix = ", ".join(f"{lang} {n}" for lang, n in langs.head(4).items())
                print(f"      {names[t][:45]:45s} n={count:3d}  ({lang_mix})")

            # ── Save BERTopic HTML charts ───────────────────────────
            charts_dir = os.path.join(args.corpus, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            for chart, make in [
                ("bertopic_barchart", lambda: topic_model.visualize_barchart(
                    top_n_topics=10, custom_labels=True)),
                ("bertopic_map", lambda: topic_model.visualize_topics(custom_labels=True)),
                ("bertopic_heatmap", lambda: topic_model.visualize_heatmap(custom_labels=True)),
            ]:
                try:
                    make().write_html(os.path.join(charts_dir, f"{chart}.html"))
                    print(f"   {charts_dir}/{chart}.html")
                except Exception:
                    pass  # The 2D map needs more themes than some runs have

        else:
            print(f"   Need 10+ documents for BERTopic (have {len(docs)})")

    except Exception as e:
        # No BERTopic or failed download: fall back to sklearn LDA
        print(f"   BERTopic unavailable ({type(e).__name__}: {str(e)[:150]})")
        print("   Falling back to sklearn LDA...")

        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            n_topics = min(5, max(3, len(df) // 10))

            cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english",
                                 max_features=1000)
            dtm = cv.fit_transform(df["content"])

            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(dtm)

            names = cv.get_feature_names_out()
            for i, w in enumerate(lda.components_):
                top = [names[j] for j in w.argsort()[-8:]]
                print(f"      Topic {i+1}: {', '.join(top)}")
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════
    # CHI-SQUARED: sentiment × source quality
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'─'*40}\nCHI-SQUARED\n{'─'*40}")

    try:
        from scipy.stats import chi2_contingency

        ct = pd.crosstab(df["quality"], df["sent_label"])

        if ct.shape[0] >= 2 and ct.shape[1] >= 2:

            chi2, p, dof, _ = chi2_contingency(ct)

            n = ct.sum().sum()
            k = min(ct.shape) - 1
            v = (chi2 / (n * k)) ** 0.5 if k > 0 and n > 0 else 0

            print(f"   χ² = {chi2:.2f}, p = {p:.4f}, Cramér's V = {v:.3f}")

            if p < 0.05:
                print("   Sentiment IS associated with source type")
            else:
                print("   No significant association")

    except ImportError:
        print("   pip install scipy")

    # ════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ════════════════════════════════════════════════════════════════

    df.to_csv(os.path.join(args.corpus, "corpus_analyzed.csv"), index=False)

    results = {
        "n_documents": len(df),
        "sentiment_by_quality": df.groupby("quality")["sentiment"].mean().to_dict(),
        "sentiment_by_lang": df.groupby("lang")["sentiment"].mean().to_dict(),
        "sent_distribution": df["sent_label"].value_counts().to_dict(),
    }

    if "topic_zs" in df.columns:
        results["zero_shot_topics"] = df["topic_zs"].value_counts().to_dict()

    if "theme" in df.columns:
        themed = df[df["bertopic"] != -1]
        results["themes"] = themed["theme"].value_counts().to_dict()
        results["theme_languages"] = {
            theme: group["lang"].value_counts().to_dict()
            for theme, group in themed.groupby("theme")}

    with open(os.path.join(args.corpus, "analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Markdown report, themes as plain lines ──────────────────────
    theme_lines = "".join(
        f"- {theme}: {n} sources ("
        + ", ".join(f"{lang} {k}" for lang, k in results["theme_languages"][theme].items()) + ")\n"
        for theme, n in results.get("themes", {}).items())
    pad("analyze_report",
        f"# NLP Analysis\n\nDocuments: {len(df)}\n\n"
        + (f"## Themes (BERTopic) and the languages discussing them\n\n{theme_lines}\n"
           if theme_lines else "")
        + json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}\nANALYSIS COMPLETE\n{'='*60}")
