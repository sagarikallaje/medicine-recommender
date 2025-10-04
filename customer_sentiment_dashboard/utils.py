from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Lazy imports for heavy libraries
try:
    import spacy
    from spacy.language import Language
    from spacy.cli import download as spacy_download
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
    Language = None  # type: ignore
    spacy_download = None  # type: ignore

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore


EMOJI_PATTERN = re.compile(
    "["  # Common emoji/codepoint ranges
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & map
    u"\U0001F1E0-\U0001F1FF"  # Flags
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
    u"\U00002600-\U000026FF"  # Misc symbols
    "]+",
    flags=re.UNICODE,
)

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass
class SentimentResult:
    label: str
    score: float


@lru_cache(maxsize=1)
def get_spacy_model() -> "Language":
    """Load spaCy English model, downloading on first use if missing.

    Returns:
        Loaded spaCy Language pipeline with tagging+lemmatization.
    """
    if spacy is None:
        raise RuntimeError("spaCy is not installed. Please install it from requirements.txt")
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name, exclude=["ner", "parser"])
    except Exception:
        # Attempt to download model if not present
        if spacy_download is None:
            raise
        spacy_download(model_name)
        nlp = spacy.load(model_name, exclude=["ner", "parser"])
    return nlp


def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)


def basic_cleanup(text: str) -> str:
    """Apply basic, non-linguistic cleanup: lowercase, remove emojis/punct, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_emojis(text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def lemmatize_and_filter(texts: Sequence[str], remove_stopwords: bool = True) -> List[str]:
    """Tokenize and lemmatize using spaCy; remove stopwords, punctuation, spaces.

    Args:
        texts: Input texts (already lowercased/cleaned preferred but not required)
        remove_stopwords: Whether to drop stopwords
    Returns:
        List of processed texts joined by spaces.
    """
    nlp = get_spacy_model()
    processed: List[str] = []
    for doc in nlp.pipe(texts, batch_size=256, disable=["ner", "parser"], n_process=1):
        tokens: List[str] = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue
            lemma = token.lemma_.strip()
            if not lemma:
                continue
            if lemma == "-PRON-":  # legacy safeguard
                lemma = token.text
            # Keep alphabetic and numeric tokens; drop residual punctuation
            if not (token.is_alpha or token.is_digit or lemma.isalnum()):
                continue
            tokens.append(lemma.lower())
        processed.append(" ".join(tokens))
    return processed


def preprocess_texts(texts: Sequence[str], apply_lemmatization: bool = True) -> List[str]:
    base_cleaned = [basic_cleanup(t) for t in texts]
    if apply_lemmatization:
        return lemmatize_and_filter(base_cleaned, remove_stopwords=True)
    return base_cleaned


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    if pipeline is None:
        raise RuntimeError("transformers.pipeline is not available. Install transformers & torch.")
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )


def classify_sentiments(
    texts: Sequence[str], neutral_threshold: float = 0.6, batch_size: int = 32
) -> List[SentimentResult]:
    """Run HF sentiment on a batch of texts, mapping low-confidence results to Neutral.

    Args:
        texts: input strings
        neutral_threshold: max probability below which we call the result Neutral
        batch_size: batch size for the pipeline
    Returns:
        List of SentimentResult with label in {Positive, Negative, Neutral} and score
    """
    if len(texts) == 0:
        return []

    clf = get_sentiment_pipeline()
    results: List[SentimentResult] = []

    # The pipeline supports batching when given a list
    raw_outputs = clf(list(texts), batch_size=batch_size)
    for out in raw_outputs:
        label_raw: str = out.get("label", "")
        score: float = float(out.get("score", 0.0))
        if score < float(neutral_threshold):
            label = "Neutral"
        else:
            label = "Positive" if "POS" in label_raw.upper() else "Negative"
        results.append(SentimentResult(label=label, score=score))
    return results


def analyze_reviews_dataframe(
    df: pd.DataFrame,
    review_column: str = "review",
    apply_preprocessing: bool = True,
    neutral_threshold: float = 0.6,
) -> pd.DataFrame:
    """Return a copy of df with `sentiment` and `score` columns added.

    The `review` column is preserved as-is; preprocessing is only used for word clouds.
    """
    if review_column not in df.columns:
        raise ValueError(f"Column '{review_column}' not in dataframe")

    # Prepare texts for classification (use original text for model)
    texts: List[str] = [str(x) if pd.notna(x) else "" for x in df[review_column].tolist()]

    sentiment_results = classify_sentiments(texts, neutral_threshold=neutral_threshold)
    sentiments = [r.label for r in sentiment_results]
    scores = [r.score for r in sentiment_results]

    result_df = df.copy()
    result_df["sentiment"] = sentiments
    result_df["score"] = scores

    # Optionally compute a processed text column for downstream word clouds
    if apply_preprocessing:
        try:
            result_df["processed_text"] = preprocess_texts(texts, apply_lemmatization=True)
        except Exception:
            # If spaCy fails for any reason, fall back to basic cleanup
            result_df["processed_text"] = preprocess_texts(texts, apply_lemmatization=False)

    return result_df


def generate_wordcloud_image(
    texts: Sequence[str],
    width: int = 900,
    height: int = 500,
    background_color: str = "white",
    max_words: int = 200,
):
    """Create a PIL image for the given texts using WordCloud."""
    # Join texts with spaces; texts should already be preprocessed
    joined = " ".join([t for t in texts if isinstance(t, str)])
    if not joined.strip():
        joined = "no data"
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        collocations=False,
    ).generate(joined)
    return wc.to_image()


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of a datetime-like column in df."""
    # Prefer common names
    preferred = [
        "timestamp",
        "time",
        "date",
        "created",
        "created_at",
        "review_time",
        "review_date",
    ]
    for name in preferred:
        if name in df.columns:
            return name
    # Fallback: check for columns that can be parsed to datetime
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(10)
        try:
            pd.to_datetime(sample, errors="raise")
            return col
        except Exception:
            continue
    return None


__all__ = [
    "get_spacy_model",
    "basic_cleanup",
    "preprocess_texts",
    "classify_sentiments",
    "analyze_reviews_dataframe",
    "generate_wordcloud_image",
    "detect_datetime_column",
    "SentimentResult",
]