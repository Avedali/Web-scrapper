"""
analyze.py: Mode 2a, NLP analysis on the collected corpus.
===========================================================
Run: python agent.py analyze
Or:  launcher.py → option 2

This mode reads the raw sources that explore gathered and runs four
types of text analysis on them with Hugging Face (HF) models:

  1. SENTIMENT ANALYSIS: scores each source as positive, negative or
     neutral about AI coding assistants. This lets you compare vendor
     sources with developer forums, and one language with another.

  2. ZERO-SHOT CLASSIFICATION: sorts each source into topics such as
     "productivity", "workforce anxiety" and "security risks" without
     training a model. You define the categories and the AI assigns
     documents to them.

  3. NAMED ENTITY RECOGNITION (NER): finds which companies, products
     and organizations are mentioned most. This shows whether the
     discussion is mostly about GitHub Copilot or also covers other tools.

  4. TOPIC MODELING (BERTopic): the AI finds the themes in the corpus
     without being told what to look for. It may find, say, 5 topics
     you didn't expect.

Finally, a chi-squared test checks whether sentiment is associated with
source type, for example whether vendor sources are statistically more
positive than forums.

All results are saved to scratchpad/ for stats mode and viz mode.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # Read/write structured data
import sys   # Stop with an error code when a step can't run
import os    # File/folder operations
import re    # Pattern matching (reading the AI's theme names)

from collections import defaultdict  # Auto-initializing dictionary (for counting entities)

from core import (
    pad,                   # Scratchpad saving
    call_ai,               # AI calls (naming themes)
    build_evidence_cards,  # 1-2 sentence English summary per source
    NO_EVIDENCE,           # Card text for sources with nothing usable
)

# How much of each source the language models read. None = all of it.
# (The models split long text into chunks, so length only costs time.
# If analyze is too slow on your laptop, set e.g. 10000. Text past the
# limit is then ignored, and that text may hold findings.)
ANALYZE_CHARS = None

# On an Apple Silicon Mac, let PyTorch run the few operations the Mac GPU
# can't do on the CPU instead of failing (must be set before torch loads)
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
    """
    Builds a Hugging Face pipeline on the fastest device that works. Apple
    Silicon Macs have a GPU that PyTorch can use ("mps"), several times
    faster than the CPU for these models. The GPU is tried on a sample
    first (`test(pipe)`). If it fails mid-run, the pipeline moves to the
    CPU, so a GPU problem cannot silently produce wrong results. Other
    computers use the default device, as before.
    """
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
# TEXT CHUNKING: lets models read every part of a long document
# ════════════════════════════════════════════════════════════════════════
#
# HF transformer models accept at most 512 "tokens" of input
# (about 400 words / 2000 characters). A 3000-character source
# would lose everything after the first ~2000 characters.
#
# Chunking fixes this: the text is split into overlapping pieces,
# the model runs on each piece, and the results are combined.
# "Overlapping" means each chunk shares ~50 characters with the next
# one, so sentences are not cut in half at chunk edges.
#
# The cost is speed: a long source becomes 3-4 model calls
# instead of 1. For 80 sources, analyze mode takes 3-4× longer.

def _progress(label, total):
    """
    Return a tick() to call once per document: it prints a progress line
    every ~10% with the time left, so a long step (NER takes ~40 minutes
    on a laptop for ~300 sources) does not look frozen.
    """
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
    """
    Split a text into overlapping chunks of roughly chunk_size characters.

    Parameters:
        text:       the full text to split
        chunk_size: how many characters per chunk (default 450, safely
                    under the 512-token limit since 1 token ≈ 4 chars)
        overlap:    how many characters each chunk shares with the next
                    (prevents cutting sentences in half)

    Returns a list of text chunks. If the text is shorter than one
    chunk, returns a single-element list with the whole text.

    Example:
        _chunk_text("a" * 1000, chunk_size=400, overlap=50)
        → ["a"*400, "a"*400, "a"*300]  (with 50-char overlaps)
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else [""]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        # Stop once this chunk reaches the end of the text. Without this,
        # the loop would add one more tiny chunk that sits entirely inside
        # the overlap of this one, double-counting it in averages/votes.
        if end >= len(text):
            break

        # Move forward by (chunk_size - overlap) so chunks overlap.
        # With chunk_size=450 and overlap=50, each step moves 400 chars.
        start += chunk_size - overlap

    return chunks


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def run_analyze(args):
    """
    Run all NLP analyses on the corpus from explore mode.
    """

    # ── Load the corpus ─────────────────────────────────────────────
    # The corpus was saved by explore mode as a JSON file: a list of
    # dictionaries, each containing a source's title, content, language,
    # quality tier, etc.
    corpus_path = os.path.join(args.corpus, "corpus.json")

    if not os.path.exists(corpus_path):
        sys.exit("No corpus. Run 'python agent.py explore' first.")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)  # Load JSON → Python list of dictionaries

    if not corpus:
        sys.exit("Corpus is empty. Re-run 'python agent.py explore'.")

    print("=" * 60)
    print(f"MODE 2a: ANALYZE ({len(corpus)} documents)")
    print("=" * 60)

    # ── Check that pandas is available ──────────────────────────────
    # pandas is the standard Python library for working with data tables.
    # A "DataFrame" is a table in code: rows and columns, with a name
    # for each column.
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    # Convert the list of dictionaries into a DataFrame (a data table).
    # Each dictionary becomes one row. The keys become column names,
    # such as "title", "content", "quality" and "lang".
    df = pd.DataFrame(corpus)

    # The language models below read each source's whole text, unless
    # ANALYZE_CHARS (set at the top of this file) limits it
    df["content"] = df["content"].fillna("")
    if ANALYZE_CHARS:
        df["content"] = df["content"].str.slice(0, ANALYZE_CHARS)

    # ── Evidence cards: a 1-2 sentence English summary per source ───
    # (see core.build_evidence_cards). Used for zero-shot topics and
    # themes below. They are cached, so synthesize later reuses them
    # without new AI calls.
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
    # ANALYSIS 1: SENTIMENT (Hugging Face transformer)
    # ════════════════════════════════════════════════════════════════
    #
    # Sentiment analysis scores whether a text is positive or negative.
    #
    # Old method (TextBlob): counts positive and negative words from a word
    #   list. "good" = +1, "bad" = -1, and the counts are added up. It is
    #   simple but ignores context.
    #   "This code is sick!" → scores as negative (the medical meaning).
    #
    # New method (Hugging Face transformer): a neural network trained on
    #   millions of examples that takes context into account.
    #   "This code is sick!" → scores as positive (slang for "excellent").
    #
    # The model used: nlptown/bert-base-multilingual-uncased-sentiment
    #   - bert-base = a standard BERT model (a balance of speed and quality)
    #   - multilingual = trained on text in many languages, English included
    #   - uncased = treats "Good" and "good" the same
    #   - sentiment = fine-tuned to predict sentiment (1-5 stars)
    #
    # Why multilingual: the corpus has sources in 9 languages. An English-only
    # model (such as distilbert-sst-2) gives meaningless scores on Japanese,
    # Chinese, Korean, Russian and other languages: the numbers carry no
    # information about the text at all. That would undermine the
    # multilingual doctrine: the analysis would conclude "Japanese sources are
    # more negative" when the real cause is that the English model can't read
    # Japanese. A multilingual model reads all these languages directly.

    print(f"\n{'─'*40}\nSENTIMENT (Hugging Face -- multilingual)\n{'─'*40}")

    try:
        # A Hugging Face "pipeline" downloads a pretrained model and returns
        # a function you can call: result = pipe("some text")
        # (_make_pipeline, above, also picks the fastest device)

        # Create a multilingual sentiment analysis pipeline.
        # This model outputs 1-5 stars, which are converted to a -1 to +1 scale.
        # The first run downloads the model (~500MB). Later runs load it
        # from the local cache in seconds.
        # truncation=True: cut long texts to fit the model's max input (512 tokens)
        # max_length=512: the model can only read 512 "tokens" (roughly words) at once
        sentiment_pipe = _make_pipeline(
            "sentiment-analysis",
            test=lambda pipe: pipe("This tool saves me time."),
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512,
        )

        def get_sentiment(text):
            """
            Analyze one text and return (score, label).

            Chunked: splits the full text into overlapping pieces,
            runs the model on each piece, and averages the scores.
            A 3000-character source becomes ~7 chunks of 450 chars each,
            so the model reads every part of the source.

            The multilingual model returns star ratings like:
              {"label": "5 stars", "score": 0.85}

            The 1-5 stars are converted to a -1 to +1 scale:
              1 star  → -1.0 (most negative)
              3 stars →  0.0 (neutral)
              5 stars → +1.0 (most positive)

            Then a label is assigned: POSITIVE (4-5), NEGATIVE (1-2), NEUTRAL (3).
            """
            try:
                # Split the text into overlapping chunks so the model
                # reads the whole document.
                chunks = _chunk_text(text)
                scores = []

                for chunk in chunks:
                    # Run the model on this chunk
                    # [0] because the pipeline returns a list (one result per input)
                    r = sentiment_pipe(chunk)[0]

                    # Extract the star number from the label ("5 stars" → 5)
                    stars = int(r["label"][0])

                    # Convert 1-5 stars to -1 to +1 scale
                    scores.append((stars - 3) / 2.0)

                # Average the scores across all chunks.
                # sum(scores) / len(scores) = the mean.
                # This gives the overall sentiment of the whole document.
                avg_score = sum(scores) / len(scores) if scores else 0.0

                # Assign a human-readable label based on the average
                if avg_score > 0.15:
                    label = "POSITIVE"
                elif avg_score < -0.15:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"

                return avg_score, label
            except Exception:
                return 0.0, "NEUTRAL"  # If anything goes wrong, default to neutral

        # .apply() runs a function on every row of a column.
        # Here it scores the sentiment of every source's content.
        # The result is a column of (score, label) tuples.
        tick = _progress("sentiment", len(df))

        def sentiment_with_progress(text):
            result = get_sentiment(text)
            tick()   # Print progress every ~10%
            return result

        sentiments = df["content"].apply(sentiment_with_progress)

        # Split the (score, label) tuples into two separate columns.
        # lambda x: x[0] means "take the first element of the tuple" (the score)
        # lambda x: x[1] means "take the second element" (the label)
        df["sentiment"] = sentiments.apply(lambda x: x[0])     # Numeric score
        df["sent_label"] = sentiments.apply(lambda x: x[1])    # POSITIVE/NEGATIVE

        # ── Group by source type and show average sentiment ─────────
        # .groupby("quality") splits the data by quality tier
        # ["sentiment"].agg(["mean", "count"]) calculates the average
        # sentiment and count for each tier
        sent_by_quality = df.groupby("quality")["sentiment"].agg(["mean", "count"])
        print("\n   By source type:")
        for q, row in sent_by_quality.iterrows():
            # q = quality tier name, row["mean"] = average sentiment
            # :30s = pad the string to 30 characters (for alignment)
            # :+.3f = show sign (+/-) and 3 decimal places
            print(f"      {q:30s}  {row['mean']:+.3f} (n={int(row['count'])})")

        # Same thing but grouped by source language
        sent_by_lang = df.groupby("lang")["sentiment"].agg(["mean", "count"])
        print("\n   By language:")
        for lang, row in sent_by_lang.iterrows():
            print(f"      {lang:5s}  {row['mean']:+.3f} (n={int(row['count'])})")

    except Exception as e:
        # Hugging Face is not installed, or the model download failed
        # (no internet, disk full, HF hub down). In both cases → TextBlob.
        # The code catches Exception instead of only ImportError because
        # pipeline() downloads the model on the first run, and that
        # download can fail with errors that are not import errors.
        print(f"   HF sentiment unavailable ({type(e).__name__}: {str(e)[:150]})")
        print("   Falling back to TextBlob...")
        try:
            from textblob import TextBlob
            # TextBlob.sentiment.polarity returns a number from -1 to +1
            df["sentiment"] = df["content"].apply(lambda t: TextBlob(t).sentiment.polarity)
            df["sent_label"] = df["sentiment"].apply(
                lambda s: "POSITIVE" if s > 0.05 else ("NEGATIVE" if s < -0.05 else "NEUTRAL")
            )
        except ImportError:
            print("   No sentiment library available")
            df["sentiment"] = 0.0       # Fill with zeros
            df["sent_label"] = "UNKNOWN"

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 2: ZERO-SHOT CLASSIFICATION (Hugging Face)
    # ════════════════════════════════════════════════════════════════
    #
    # This HF feature is especially useful for research. "Zero-shot" means
    # the model was never trained on your specific categories. You define
    # them when the code runs and it classifies the text anyway.
    #
    # How it works: the model is trained on "natural language inference"
    # (NLI). Given two sentences, it decides whether the second follows
    # from the first, contradicts it, or is neutral.
    #
    # To classify, it turns each category into a hypothesis:
    #   Text: "GitHub reports 40% of code is now AI-generated"
    #   Hypothesis 1: "This text is about developer productivity" → score 0.8
    #   Hypothesis 2: "This text is about workforce displacement" → score 0.2
    #   → Winner: "developer productivity"
    #
    # You can change the categories to anything, with no retraining.
    #
    # Why multilingual: same reason as for sentiment. facebook/bart-large-mnli
    # is English-only and gives meaningless results on Japanese, Chinese and
    # Korean sources. joeddav/xlm-roberta-large-xnli was trained on XNLI
    # (Cross-lingual NLI), which covers 15 languages. It can put a Japanese
    # article about an engineer shortage into "workforce displacement"
    # because it reads Japanese.

    print("\n" + '─'*40 + "\n ZERO-SHOT CLASSIFICATION (HF -- multilingual)\n" + '─'*40)

    try:
        # Create a multilingual zero-shot classification pipeline.
        # xlm-roberta = cross-lingual RoBERTa (reads 100+ languages)
        # xnli = cross-lingual NLI (trained on 15 languages)
        zs_pipe = _make_pipeline(
            "zero-shot-classification",
            test=lambda pipe: pipe("This tool saves me time.", candidate_labels=["time", "cost"]),
            model="joeddav/xlm-roberta-large-xnli",  # Multilingual XNLI
            truncation=True,
        )

        # The categories to sort sources into.
        # Change them to whatever fits your research.
        # The model has never seen these exact labels. It picks the best
        # match from its general knowledge of language.
        categories = [
            "developer productivity and efficiency",
            "workforce displacement and job loss concerns",
            "code quality and security risks",
            "industry adoption and market trends",
            "regulation and ethics",
            "developer experience and satisfaction",
        ]

        def classify(text):
            """
            Classify one text into the best-matching category.

            Chunked: splits the text into pieces, classifies each piece,
            and picks the category that wins the most chunks.
            This stops a long article about "workforce displacement"
            from being classified as "productivity" only because its
            first paragraph mentions efficiency.
            """
            try:
                chunks = _chunk_text(text)
                votes = defaultdict(int)  # Count how many chunks vote for each category

                for chunk in chunks:
                    # Classify this chunk
                    r = zs_pipe(chunk, candidate_labels=categories)
                    # r["labels"][0] = the top category for this chunk
                    winner = r["labels"][0]
                    votes[winner] += 1  # This category gets one vote

                # Return the category with the most votes.
                # max(votes, key=votes.get) finds the key with the highest value.
                # If votes = {"productivity": 2, "workforce": 5}, returns "workforce".
                return max(votes, key=votes.get) if votes else "unclassified"
            except Exception:
                return "unclassified"

        print(f"   Classifying {len(df)} documents into {len(categories)} categories...")

        # Apply the classifier to every document's content.
        # This creates a new column "topic_zs" (zs = zero-shot).
        # When a source has an evidence card, the card is classified: a single
        # English summary of what the source says. That replaces ~7 chunks of
        # raw page text per source and is about 7x faster on a laptop CPU.
        tick = _progress("zero-shot", len(df))

        def classify_with_progress(text):
            result = classify(text)
            tick()   # Print progress every ~10%
            return result

        df["topic_zs"] = df.apply(card_or_text, axis=1).apply(classify_with_progress)

        # Show how many documents fell into each category
        # .value_counts() counts occurrences of each unique value
        topic_counts = df["topic_zs"].value_counts()
        print("\n   Topic distribution:")
        for topic, count in topic_counts.items():
            print(f"      {topic:45s}  {count}")

        # ── Cross-tabulation: topic × language ──────────────────────
        # A cross-tab shows how two categorical variables relate.
        # It is a table where rows = languages, columns = topics,
        # and each cell = "how many documents of this language are in this topic"
        #
        # It can show patterns such as: "Japanese sources cluster in 'workforce'
        # while English sources cluster in 'productivity'"
        ct = pd.crosstab(df["lang"], df["topic_zs"])

        # .shape = (number of rows, number of columns)
        # The table needs at least 2 rows (2 languages) to compare anything
        if ct.shape[0] >= 2:
            print("\n   Topic × Language cross-tab:")
            print(ct.to_string())  # .to_string() renders the table as text

    except Exception as e:
        # Covers both a missing install and a failed model download.
        # The message shows the real reason. Often the cause is a missing
        # helper package (e.g. sentencepiece) and transformers itself is fine.
        print(f"   zero-shot unavailable ({type(e).__name__}: {str(e)[:150]})")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 3: NAMED ENTITY RECOGNITION (NER)
    # ════════════════════════════════════════════════════════════════
    #
    # NER finds and labels named things in text:
    #   - ORG = organizations (Microsoft, GitHub, OECD)
    #   - PER = people (Satya Nadella, Linus Torvalds)
    #   - LOC = locations (Silicon Valley, Brussels)
    #
    # The code counts which organizations appear most across all sources.
    # This shows who dominates the discussion: Copilot specifically, or
    # AI coding tools in general.
    #
    # The model: Davlan/xlm-roberta-large-ner-hrl
    #   - xlm-roberta-large = cross-lingual RoBERTa (reads 100+ languages)
    #   - ner-hrl = fine-tuned for NER on high-resource languages
    #     (English, French, German, Spanish, Dutch, Chinese, Arabic, etc.)
    #
    # Why multilingual: the corpus has Japanese, Chinese, Korean and Russian
    # sources. An English-only NER model can't extract organization names
    # from non-Latin scripts. It would miss 株式会社GitHub (GitHub Inc. in
    # Japanese) or 微软 (Microsoft in Chinese). This multilingual model
    # reads names in all the source languages.

    print(f"\n{'─'*40}\nNAMED ENTITY RECOGNITION (HF -- multilingual)\n{'─'*40}")

    try:
        # aggregation_strategy="simple" merges multi-word entities.
        # Without it, "New" and "York" would be two separate entities.
        # With it, "New York" is one entity.
        ner_pipe = _make_pipeline(
            "ner",
            test=lambda pipe: pipe("GitHub and Google work with developers in Paris."),
            aggregation_strategy="simple",
            model="Davlan/xlm-roberta-large-ner-hrl",  # Multilingual NER
            # No truncation=True here: the NER pipeline rejects that argument
            # (TypeError). Our 450-character chunks already fit in 512 tokens.
        )

        # Count how many times each organization appears across all sources
        # defaultdict(int) auto-initializes to 0 for new keys
        all_entities = defaultdict(int)

        # .iterrows() loops through the DataFrame row by row
        # _ means "we don't need the row index, throw it away"
        tick = _progress("NER", len(df))
        for _, row in df.iterrows():
            tick()
            try:
                # Chunked: split the source into pieces and run NER on each.
                # Without chunking, an organization mentioned only on page 3
                # of a long article would be missed.
                chunks = _chunk_text(row["content"])

                for chunk in chunks:
                    entities = ner_pipe(chunk)

                    for e in entities:
                        # Keep organizations (ORG) and drop people and locations.
                        # Keep only high-confidence predictions (score > 0.8).
                        # (The model sometimes returns an empty "word" or bare
                        # punctuation such as "’" or "。". These are skipped,
                        # because otherwise they would rank as top organizations.)
                        word = e["word"].strip()
                        is_name = any(ch.isalnum() for ch in word)
                        if e["entity_group"] == "ORG" and e["score"] > 0.8 and is_name:
                            all_entities[word] += 1
            except Exception:
                pass  # Skip sources that cause errors

        # Sort by count (most mentioned first) and take top 20
        # sorted(..., key=lambda x: -x[1]) sorts by the count (x[1])
        # in descending order (the minus sign reverses the sort)
        top_orgs = sorted(all_entities.items(), key=lambda x: -x[1])[:20]

        print("   Most mentioned organizations:")
        for org, count in top_orgs:
            print(f"      {org:30s}  {count} mentions")

    except Exception as e:
        # Covers both a missing install and a failed model download
        print(f"   NER unavailable ({type(e).__name__}: {str(e)[:150]})")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 4: THEMES (BERTopic on evidence cards)
    # ════════════════════════════════════════════════════════════════
    #
    # Topic modeling finds themes in a collection of texts without being
    # told what to look for. Example result: 8 clusters, with a description
    # of what each one is about.
    #
    # BERTopic turns each text into embeddings (lists of numbers that
    # represent its meaning) and clusters similar ones together.
    #
    # Why cluster evidence cards instead of the raw pages:
    # On raw pages in 9 languages, topics came out labelled with words like
    # "die / der / de" (German, French and Spanish filler words), and the
    # page boilerplate (menus, "h2", cookie text) diluted the meaning.
    # Evidence cards (core.build_evidence_cards) are 1-2 sentence English
    # summaries of each source's findings. Clustering them groups sources
    # by what they say, whatever language they are in. An AI call then
    # names each theme in plain English.
    #
    # The code uses k-means (a fixed number of themes) instead of BERTopic's
    # default HDBSCAN. The cards are all about AI coding tools, so HDBSCAN
    # put ~90% of them into one topic, or left ~45% as "outliers".
    #
    # Writing the cards needs an API key. Without one, the code clusters the
    # raw pages and labels topics with meaning-based keywords.
    #
    # Fallback: if BERTopic isn't installed, the code uses LDA (Latent
    # Dirichlet Allocation), an older method that works on word counts.
    # LDA is English-only and needs more documents to work well.

    print(f"\n{'─'*40}\nTHEMES (BERTopic)\n{'─'*40}")

    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired  # Meaning-based keywords
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer

        # ── What to cluster: evidence cards if we have them, raw pages if not ─
        if cards:
            # Sources with nothing usable don't get a theme
            keep = [i for i, idx in enumerate(df["idx"])
                    if idx in cards and not cards[idx].startswith(NO_EVIDENCE)]
            docs = [cards[df["idx"].iloc[i]] for i in keep]
        else:
            keep = list(range(len(df)))
            docs = df["content"].tolist()

        if len(docs) >= 10:
            # About one theme per 12 sources, between 5 and 10 themes
            n_themes = max(5, min(10, len(docs) // 12))

            topic_model = BERTopic(
                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                hdbscan_model=KMeans(n_clusters=n_themes, random_state=42, n_init=10),
                # Keywords: skip English filler, allow 2-word phrases, and
                # pick words by meaning (KeyBERTInspired) rather than raw counts
                vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                representation_model=KeyBERTInspired(),
                verbose=False,
            )

            # fit_transform does two things:
            #   fit = learn the themes from the texts
            #   transform = assign each text to a theme
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
            # A theme without an AI name is named by its top 4 keywords
            for t in set(topics):
                if t not in names:
                    names[t] = ", ".join(w for w, _ in topic_model.get_topic(t)[:4])
            topic_model.set_topic_labels(names)

            # ── Store the theme of each source (-1 = no theme) ──────
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

            # ── Save BERTopic's built-in visualizations ─────────────
            # BERTopic generates its own interactive plotly charts.
            # We save them as HTML files you can open in a browser.
            # custom_labels=True shows the English theme names.
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
                    pass  # e.g. the 2D map needs more themes than we have

        else:
            print(f"   Need 10+ documents for BERTopic (have {len(docs)})")

    except Exception as e:
        # BERTopic is not installed or its embedding-model download failed
        # → fall back to LDA (older, simpler, needs no downloads)
        print(f"   BERTopic unavailable ({type(e).__name__}: {str(e)[:150]})")
        print("   Falling back to sklearn LDA...")

        try:
            # CountVectorizer turns text into a word-count matrix.
            # Each document becomes a row, each unique word becomes a column,
            # and each cell = how many times that word appears in that document.
            from sklearn.feature_extraction.text import CountVectorizer
            # LDA finds topics by looking for groups of words that tend to
            # appear together across documents.
            from sklearn.decomposition import LatentDirichletAllocation

            # Choose number of topics: at least 3, at most 5,
            # roughly 1 topic per 10 documents.
            # min() returns the smaller value, max() the larger.
            # len(df) // 10 is integer division (80 docs → 8 → capped at 5)
            n_topics = min(5, max(3, len(df) // 10))

            # Build the word-count matrix.
            # max_df=0.9: ignore words that appear in >90% of docs (too common)
            # min_df=2: ignore words that appear in <2 docs (too rare)
            # stop_words="english": ignore common words like "the", "is", "and"
            # max_features=1000: only keep the top 1000 most common words
            cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english",
                                 max_features=1000)
            dtm = cv.fit_transform(df["content"])  # dtm = document-term matrix

            # Fit the LDA model
            # n_components = number of topics to discover
            # random_state=42: makes results reproducible (same input → same output)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(dtm)

            # Show the top 8 words for each topic
            names = cv.get_feature_names_out()  # List of all words in the vocabulary
            for i, w in enumerate(lda.components_):
                # w.argsort() returns indices that would sort the weights
                # [-8:] takes the last 8 (highest weights = most important words)
                top = [names[j] for j in w.argsort()[-8:]]
                print(f"      Topic {i+1}: {', '.join(top)}")
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════
    # CHI-SQUARED TEST: sentiment × source quality
    # ════════════════════════════════════════════════════════════════
    #
    # The chi-squared test checks whether sentiment is associated with
    # source type or independent of it.
    #
    # Null hypothesis (H₀): the sentiment distribution is the same for
    # every source type. (Vendor sources are as likely to be negative
    # as forum sources.)
    #
    # If p < 0.05, H₀ is rejected → sentiment is associated with source
    # type. (For example, vendor sources are systematically more positive.)
    #
    # Cramér's V measures how strong the association is (0 to 1):
    #   V < 0.1 = negligible, V < 0.3 = weak, V < 0.5 = moderate, V ≥ 0.5 = strong

    print(f"\n{'─'*40}\nCHI-SQUARED\n{'─'*40}")

    try:
        from scipy.stats import chi2_contingency

        # Build a contingency table (cross-tab):
        # rows = quality tiers, columns = sentiment labels (POS/NEG/NEUTRAL)
        # each cell = count of sources in that combination
        ct = pd.crosstab(df["quality"], df["sent_label"])

        # Need at least 2 rows and 2 columns for the test to work
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:

            # Run the chi-squared test.
            # Returns: chi2 (test statistic), p (p-value), dof (degrees of freedom),
            # and expected frequencies (which we don't use, hence _)
            chi2, p, dof, _ = chi2_contingency(ct)

            # Calculate Cramér's V (effect size)
            n = ct.sum().sum()      # Total number of observations
            k = min(ct.shape) - 1   # min(rows, cols) - 1
            v = (chi2 / (n * k)) ** 0.5 if k > 0 and n > 0 else 0
            # ** 0.5 = square root (x ** 0.5 = √x)

            print(f"   χ² = {chi2:.2f}, p = {p:.4f}, Cramér's V = {v:.3f}")

            if p < 0.05:
                print("   Sentiment IS associated with source type")
            else:
                print("   No significant association")

    except ImportError:
        print("   pip install scipy")

    # ════════════════════════════════════════════════════════════════
    # SAVE RESULTS for downstream modes (stats, viz, synthesize)
    # ════════════════════════════════════════════════════════════════

    # Save the enriched DataFrame as CSV (stats and viz will read this).
    # index=False means "don't save the row numbers as a column"
    df.to_csv(os.path.join(args.corpus, "corpus_analyzed.csv"), index=False)

    # Save a summary of results as JSON (for synthesize to read)
    results = {
        "n_documents": len(df),
        # .groupby("quality")["sentiment"].mean() → average sentiment per quality tier
        # .to_dict() → convert to a plain dictionary (JSON-compatible)
        "sentiment_by_quality": df.groupby("quality")["sentiment"].mean().to_dict(),
        "sentiment_by_lang": df.groupby("lang")["sentiment"].mean().to_dict(),
        "sent_distribution": df["sent_label"].value_counts().to_dict(),
    }

    # Add zero-shot topic counts if we ran that analysis
    if "topic_zs" in df.columns:
        results["zero_shot_topics"] = df["topic_zs"].value_counts().to_dict()

    # Add the BERTopic themes (with their English names) if we found any
    if "theme" in df.columns:
        themed = df[df["bertopic"] != -1]
        results["themes"] = themed["theme"].value_counts().to_dict()
        # Which languages discuss each theme (so synthesize/ask can see it)
        results["theme_languages"] = {
            theme: group["lang"].value_counts().to_dict()
            for theme, group in themed.groupby("theme")}

    with open(os.path.join(args.corpus, "analysis_results.json"), "w") as f:
        # default=str tells json.dump to convert types it can't store to strings
        # (some pandas types aren't JSON-compatible by default)
        json.dump(results, f, indent=2, default=str)

    # Save a markdown report to the scratchpad.
    # The themes are also written as plain sentences, which people (and
    # the ask agent) read more easily than the nested JSON below
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
