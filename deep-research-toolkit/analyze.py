"""
analyze.py -- Mode 2a: NLP analysis on the collected corpus.
=============================================================
Run: python agent.py analyze
Or:  launcher.py → option 2

This mode reads the raw sources that explore gathered, and runs
FOUR types of text analysis on them using Hugging Face (HF) models:

  1. SENTIMENT ANALYSIS -- is each source positive, negative, or neutral
     about AI coding assistants? Do vendor sources feel different from
     developer forums? Do different languages have different vibes?

  2. ZERO-SHOT CLASSIFICATION -- classify each source into topics like
     "productivity", "workforce anxiety", "security risks" WITHOUT
     training any model. You just define the categories and the AI
     sorts documents into them.

  3. NAMED ENTITY RECOGNITION (NER) -- extract which companies, products,
     and organizations are mentioned most. Is the conversation mostly
     about GitHub Copilot, or are other tools getting attention too?

  4. TOPIC MODELING (BERTopic) -- let the AI discover what themes exist
     in the corpus, without you telling it what to look for. It might
     find 5 topics you didn't expect.

Finally, a chi-squared test checks: is sentiment ASSOCIATED with source
type? (Are vendor sources statistically more positive than forums?)

All results are saved to scratchpad/ for stats mode and viz mode.
"""

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════

import json  # Read/write structured data
import os    # File/folder operations

from collections import defaultdict  # Auto-initializing dictionary (for counting entities)

from core import pad  # Scratchpad saving


# ════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING -- lets models read entire documents, not just page one
# ════════════════════════════════════════════════════════════════════════
#
# HF transformer models have a maximum input size of 512 "tokens"
# (roughly 400 words / 2000 characters). A 3000-character source
# would lose everything after the first ~2000 characters.
#
# CHUNKING solves this: split the text into overlapping pieces,
# run the model on each piece, and combine the results.
# "Overlapping" means each chunk shares ~50 characters with the next
# one, so we don't accidentally cut a sentence in half.
#
# The tradeoff is speed: a long source becomes 3-4 model calls
# instead of 1. For 80 sources, analyze mode takes 3-4× longer.

def _chunk_text(text, chunk_size=450, overlap=50):
    """
    Split a text into overlapping chunks of roughly chunk_size characters.

    Parameters:
        text       -- the full text to split
        chunk_size -- how many characters per chunk (default 450, safely
                     under the 512-token limit since 1 token ≈ 4 chars)
        overlap    -- how many characters of overlap between chunks
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
        # If chunk_size=450 and overlap=50, we advance 400 chars each step.
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
        print("No corpus. Run 'python agent.py explore' first.")
        return

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)  # Load JSON → Python list of dictionaries

    if not corpus:
        print("Corpus is empty. Re-run 'python agent.py explore'.")
        return

    print("=" * 60)
    print(f"MODE 2a: ANALYZE ({len(corpus)} documents)")
    print("=" * 60)

    # ── Check that pandas is available ──────────────────────────────
    # pandas is the standard Python library for working with data tables.
    # A "DataFrame" is like a spreadsheet in code: rows and columns,
    # with names for each column.
    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas")
        return

    # Convert our list of dictionaries into a DataFrame (a data table).
    # Each dictionary becomes one row. The keys become column names.
    # So we get columns like: "title", "content", "quality", "lang", etc.
    df = pd.DataFrame(corpus)

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 1: SENTIMENT (Hugging Face transformer)
    # ════════════════════════════════════════════════════════════════
    #
    # Sentiment analysis answers: "is this text positive or negative?"
    #
    # OLD way (TextBlob): counts positive/negative words from a dictionary.
    #   "good" = +1, "bad" = -1, add them up. Simple but dumb.
    #   "This code is sick!" → scores as negative (medical context).
    #
    # NEW way (Hugging Face transformer): a neural network trained on
    #   millions of examples that understands CONTEXT.
    #   "This code is sick!" → scores as positive (slang for "excellent").
    #
    # The model we use: nlptown/bert-base-multilingual-uncased-sentiment
    #   - bert-base = a standard BERT model (good balance of speed and quality)
    #   - multilingual = trained on text in MANY languages (not just English)
    #   - uncased = treats "Good" and "good" the same
    #   - sentiment = fine-tuned to predict sentiment (1-5 stars)
    #
    # WHY MULTILINGUAL? Our corpus has sources in 9 languages. An English-only
    # model (like distilbert-sst-2) produces GARBAGE scores on Japanese, Chinese,
    # Korean, Russian, etc. -- not "less accurate," genuinely meaningless numbers.
    # That would undermine the entire multilingual doctrine: we'd conclude
    # "Japanese sources are more negative" when the real finding is "the English
    # model can't read Japanese." A multilingual model reads all languages natively.

    print(f"\n{'─'*40}\nSENTIMENT (Hugging Face -- multilingual)\n{'─'*40}")

    try:
        # "pipeline" is Hugging Face's magic one-liner.
        # pipeline("task-name") downloads a pretrained model and gives you
        # a function you can call: result = pipe("some text")
        from transformers import pipeline

        # Create a MULTILINGUAL sentiment analysis pipeline.
        # This model outputs 1-5 stars. We convert to a -1 to +1 scale.
        # The first time you run this, it downloads the model (~500MB).
        # After that, it's cached locally and loads instantly.
        # truncation=True: cut long texts to fit the model's max input (512 tokens)
        # max_length=512: the model can only read 512 "tokens" (roughly words) at once
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512,
        )

        def get_sentiment(text):
            """
            Analyze one text and return (score, label).

            CHUNKED: splits the full text into overlapping pieces,
            runs the model on each piece, and AVERAGES the scores.
            A 3000-character source becomes ~7 chunks of 450 chars each.
            This means the model reads the ENTIRE source, not just the
            first page.

            The MULTILINGUAL model returns star ratings like:
              {"label": "5 stars", "score": 0.85}

            We convert 1-5 stars to a -1 to +1 scale:
              1 star  → -1.0 (very negative)
              3 stars →  0.0 (neutral)
              5 stars → +1.0 (very positive)

            And assign a label: POSITIVE (4-5), NEGATIVE (1-2), NEUTRAL (3).
            """
            try:
                # Split the text into overlapping chunks so the model
                # reads the whole document, not just the beginning.
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
                # This gives us the overall sentiment of the WHOLE document,
                # not just the intro.
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
        # So this analyzes the sentiment of every source's content.
        # The result is a column of (score, label) tuples.
        sentiments = df["content"].apply(get_sentiment)

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
        # Hugging Face not installed, OR the model download failed
        # (no internet, disk full, HF hub down). Either way → TextBlob.
        # We catch Exception (not just ImportError) because pipeline()
        # DOWNLOADS the model on first run, and that can fail in ways
        # that aren't import errors.
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
    # This is the coolest HF feature for research. "Zero-shot" means
    # the model was NOT trained on your specific categories -- you
    # define them at runtime and it classifies anyway.
    #
    # How? The model is trained on "natural language inference" (NLI) --
    # given two sentences, decide if the second FOLLOWS from the first,
    # CONTRADICTS it, or is NEUTRAL.
    #
    # To classify, it turns each category into a hypothesis:
    #   Text: "GitHub reports 40% of code is now AI-generated"
    #   Hypothesis 1: "This text is about developer productivity" → score 0.8
    #   Hypothesis 2: "This text is about workforce displacement" → score 0.2
    #   → Winner: "developer productivity"
    #
    # You can change the categories to anything you want. No retraining.
    #
    # WHY MULTILINGUAL? Same reason as sentiment: facebook/bart-large-mnli
    # is English-only and produces garbage on Japanese/Chinese/Korean sources.
    # joeddav/xlm-roberta-large-xnli was trained on XNLI (Cross-lingual NLI)
    # which covers 15 languages. It can classify a Japanese article about
    # engineer shortage into "workforce displacement" because it READS Japanese.

    print("\n" + '─'*40 + "\n ZERO-SHOT CLASSIFICATION (HF -- multilingual)\n" + '─'*40)

    try:
        from transformers import pipeline

        # Create a MULTILINGUAL zero-shot classification pipeline.
        # xlm-roberta = cross-lingual RoBERTa (reads 100+ languages)
        # xnli = cross-lingual NLI (trained on 15 languages)
        zs_pipe = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",  # Multilingual XNLI
            truncation=True,
        )

        # These are the categories we want to sort sources into.
        # You can change these to anything relevant to your research.
        # The model has never seen these exact labels before -- it figures
        # out the best match using its understanding of language.
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

            CHUNKED: splits the text into pieces, classifies each,
            and picks the category that wins the MOST chunks.
            This prevents a long article about "workforce displacement"
            from being classified as "productivity" just because its
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

        # Apply the classifier to every document's content
        # This creates a new column "topic_zs" (zs = zero-shot)
        df["topic_zs"] = df["content"].apply(classify)

        # Show how many documents fell into each category
        # .value_counts() counts occurrences of each unique value
        topic_counts = df["topic_zs"].value_counts()
        print("\n   Topic distribution:")
        for topic, count in topic_counts.items():
            print(f"      {topic:45s}  {count}")

        # ── Cross-tabulation: topic × language ──────────────────────
        # A cross-tab shows how two categorical variables relate.
        # It's a table where rows = languages, columns = topics,
        # and each cell = "how many documents of this language are in this topic"
        #
        # This reveals things like: "Japanese sources cluster in 'workforce'
        # while English sources cluster in 'productivity'"
        ct = pd.crosstab(df["lang"], df["topic_zs"])

        # .shape = (number of rows, number of columns)
        # We need at least 2 rows (2 languages) for the table to be interesting
        if ct.shape[0] >= 2:
            print("\n   Topic × Language cross-tab:")
            print(ct.to_string())  # .to_string() renders the table as text

    except Exception as e:
        # Covers both missing install AND model-download failures
        # Show the real reason -- it's often a missing helper package
        # (e.g. sentencepiece), not transformers itself.
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
    # We count which organizations appear most across all sources.
    # This tells us who dominates the discourse -- is the conversation
    # about Copilot specifically, or about AI coding tools broadly?
    #
    # The model: Davlan/xlm-roberta-large-ner-hrl
    #   - xlm-roberta-large = cross-lingual RoBERTa (reads 100+ languages)
    #   - ner-hrl = fine-tuned for NER on high-resource languages
    #     (English, French, German, Spanish, Dutch, Chinese, Arabic, etc.)
    #
    # WHY MULTILINGUAL? Our corpus has Japanese, Chinese, Korean, Russian
    # sources. An English-only NER model can't extract organization names
    # from non-Latin scripts. 株式会社GitHub (GitHub Inc. in Japanese) or
    # 微软 (Microsoft in Chinese) would be invisible to an English model.
    # This multilingual model reads names in all our source languages.

    print(f"\n{'─'*40}\nNAMED ENTITY RECOGNITION (HF -- multilingual)\n{'─'*40}")

    try:
        from transformers import pipeline

        # aggregation_strategy="simple" merges multi-word entities.
        # Without it, "New" and "York" would be two separate entities.
        # With it, "New York" is one entity.
        ner_pipe = pipeline(
            "ner",
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
        for _, row in df.iterrows():
            try:
                # CHUNKED: split the source into pieces and run NER on each.
                # An organization mentioned only on page 3 of a long article
                # would be invisible without chunking.
                chunks = _chunk_text(row["content"])

                for chunk in chunks:
                    entities = ner_pipe(chunk)

                    for e in entities:
                        # Only keep organizations (ORG), not people or locations
                        # Only keep high-confidence predictions (score > 0.8)
                        # (The model occasionally returns an empty "word" --
                        # skip it, or "" would rank as a top organization.)
                        word = e["word"].strip()
                        if e["entity_group"] == "ORG" and e["score"] > 0.8 and word:
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
        # Covers both missing install AND model-download failures
        print(f"   NER unavailable ({type(e).__name__}: {str(e)[:150]})")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 4: TOPIC MODELING (BERTopic)
    # ════════════════════════════════════════════════════════════════
    #
    # Topic modeling discovers themes in a collection of texts WITHOUT
    # you telling it what to look for. It reads all 80 documents and
    # says "I found 5 clusters -- here's what each one is about."
    #
    # BERTopic is the modern approach. It uses transformer embeddings
    # (converting each document into a list of numbers that represent
    # its meaning) and then clusters those number-lists to find groups
    # of documents that are about similar things.
    #
    # KEY ADVANTAGE: language="multilingual" means it understands
    # Japanese, French, German, etc. natively. A Japanese source about
    # "engineer shortage" and an English source about "developer hiring
    # crisis" end up in the same topic because the MEANING is similar,
    # even though the WORDS are completely different.
    #
    # FALLBACK: If BERTopic isn't installed, we use LDA (Latent Dirichlet
    # Allocation), an older method that works on word counts. LDA is
    # English-only and needs more documents to work well.

    print(f"\n{'─'*40}\nBERTOPIC (neural topic modeling)\n{'─'*40}")

    try:
        from bertopic import BERTopic

        # Convert the content column to a plain Python list
        # (BERTopic expects a list of strings, not a pandas Series)
        docs = df["content"].tolist()

        # BERTopic needs at least 10 documents to find meaningful topics
        if len(docs) >= 10:

            # Create and run the BERTopic model.
            # language="multilingual": use a model that understands many languages
            # min_topic_size=3: a topic must have at least 3 documents
            # nr_topics="auto": let BERTopic decide how many topics there are
            # verbose=False: don't print progress messages
            topic_model = BERTopic(
                language="multilingual",
                min_topic_size=3,
                nr_topics="auto",
                verbose=False,
            )

            # fit_transform does two things:
            #   fit = learn the topics from the documents
            #   transform = assign each document to a topic
            # topics = a list of topic numbers (one per document, -1 = outlier)
            # probs = confidence scores (how sure the model is about each assignment)
            topics, probs = topic_model.fit_transform(docs)

            # Add the topic assignment to our DataFrame
            df["bertopic"] = topics

            # Get a summary table of all discovered topics
            topic_info = topic_model.get_topic_info()

            # Topic -1 is the "outlier" bin (documents that don't fit any topic)
            # so we subtract 1 from the total count
            print(f"   Found {len(topic_info) - 1} topics (excluding outliers)")

            # Show the top 8 topics
            # .head(8) takes the first 8 rows of the table
            for _, row in topic_info.head(8).iterrows():
                if row["Topic"] != -1:  # Skip the outlier topic
                    # row["Name"] is auto-generated from the topic's top words
                    # row["Count"] is how many documents are in this topic
                    print(f"      Topic {row['Topic']}: {row['Name'][:60]} (n={row['Count']})")

            # ── Save BERTopic's built-in visualizations ─────────────
            # BERTopic generates its own interactive plotly charts.
            # We save them as HTML files you can open in a browser.
            charts_dir = os.path.join(args.corpus, "charts")
            os.makedirs(charts_dir, exist_ok=True)

            # Bar chart: top words for each topic
            try:
                fig = topic_model.visualize_barchart(top_n_topics=8)
                fig.write_html(os.path.join(charts_dir, "bertopic_barchart.html"))
                print(f"   {charts_dir}/bertopic_barchart.html")
            except Exception:
                pass

            # Topic map: a 2D plot where similar topics are close together
            try:
                fig = topic_model.visualize_topics()
                fig.write_html(os.path.join(charts_dir, "bertopic_map.html"))
                print(f"   {charts_dir}/bertopic_map.html")
            except Exception:
                pass

            # Heatmap: how similar each pair of topics is to each other
            try:
                fig = topic_model.visualize_heatmap()
                fig.write_html(os.path.join(charts_dir, "bertopic_heatmap.html"))
                print(f"   {charts_dir}/bertopic_heatmap.html")
            except Exception:
                pass

        else:
            print(f"   Need 10+ documents for BERTopic (have {len(docs)})")

    except Exception as e:
        # BERTopic not installed OR its embedding-model download failed
        # → fall back to LDA (older, simpler, no downloads needed)
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
    # The chi-squared test asks: "Is sentiment ASSOCIATED with source
    # type, or are they independent?"
    #
    # Null hypothesis (H₀): Sentiment distribution is the same
    # regardless of source type. (Vendor sources are just as likely
    # to be negative as forum sources.)
    #
    # If p < 0.05: we reject H₀ → sentiment IS associated with source
    # type. (Vendor sources are systematically more positive.)
    #
    # Cramér's V measures HOW STRONG the association is (0 to 1):
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

    with open(os.path.join(args.corpus, "analysis_results.json"), "w") as f:
        # default=str tells json.dump to convert any weird types to strings
        # (some pandas types aren't JSON-compatible by default)
        json.dump(results, f, indent=2, default=str)

    # Save a markdown report to the scratchpad
    pad("analyze_report",
        f"# NLP Analysis\n\nDocuments: {len(df)}\n"
        + json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}\nANALYSIS COMPLETE\n{'='*60}")
