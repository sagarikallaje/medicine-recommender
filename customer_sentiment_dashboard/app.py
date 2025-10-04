from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    analyze_reviews_dataframe,
    detect_datetime_column,
    generate_wordcloud_image,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SAMPLE_CSV_PATH = DATA_DIR / "sample_reviews.csv"

st.set_page_config(page_title="Customer Sentiment Analysis Dashboard", layout="wide")


def load_sample_dataframe() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_CSV_PATH)
    return df


def try_parse_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    try:
        return pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(df)))


@st.cache_data(show_spinner=False)
def cached_analyze(
    df: pd.DataFrame,
    review_col: str,
    neutral_threshold: float,
    apply_preprocessing: bool,
) -> pd.DataFrame:
    return analyze_reviews_dataframe(
        df=df,
        review_column=review_col,
        apply_preprocessing=apply_preprocessing,
        neutral_threshold=neutral_threshold,
    )


# Sidebar: Data source and options
with st.sidebar:
    st.title("Settings")

    source = st.radio("Data source", ["Sample", "Upload CSV"], index=0, horizontal=True)
    uploaded_df: Optional[pd.DataFrame] = None

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception:
                st.error("Failed to read CSV. Please check the file format.")

    neutral_threshold = st.slider(
        "Neutral confidence threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.01,
        help="Predictions below this confidence are labeled Neutral.",
    )

    apply_preprocessing = st.checkbox(
        "Preprocess text (clean + lemmatize)", value=True,
        help="Used for word clouds; original text is used for model inference.",
    )

    chart_type = st.selectbox("Distribution chart", ["Bar", "Pie"], index=0)

    max_examples = st.number_input(
        "Examples per sentiment", min_value=1, max_value=20, value=5, step=1
    )

    max_wc_words = st.slider("WordCloud max words", 50, 500, 200, step=10)


# Main area
st.title("Customer Sentiment Analysis Dashboard")

# Load data
if source == "Sample":
    df_raw = load_sample_dataframe()
else:
    if uploaded_df is None:
        st.info("Upload a CSV with a 'review' column to start.")
        st.stop()
    df_raw = uploaded_df

if df_raw.empty:
    st.warning("The dataset is empty.")
    st.stop()

# Let user select the review column if there are multiple text-like columns
text_like_cols = [
    c for c in df_raw.columns if df_raw[c].dtype == object or pd.api.types.is_string_dtype(df_raw[c])
]

default_review_col = "review" if "review" in df_raw.columns else (text_like_cols[0] if text_like_cols else None)
review_col = st.selectbox("Review text column", options=text_like_cols, index=(text_like_cols.index(default_review_col) if default_review_col in text_like_cols else 0))

# Optional datetime column
candidate_time_col = detect_datetime_column(df_raw) or "(none)"
time_cols = ["(none)"] + list(df_raw.columns)
if candidate_time_col != "(none)" and candidate_time_col in df_raw.columns:
    default_time_index = time_cols.index(candidate_time_col)
else:
    default_time_index = 0

time_col_choice = st.selectbox("Timestamp column (optional)", options=time_cols, index=default_time_index)

# Search / filter
search_query = st.text_input("Search reviews (keyword)", value="")

# Run analysis on demand
run_clicked = st.button("Run sentiment analysis", type="primary")

if not run_clicked and "analysis_ready" not in st.session_state:
    st.info("Click 'Run sentiment analysis' to begin.")
    st.stop()

# Perform analysis (cache-aware)
with st.spinner("Analyzing sentiments..."):
    df_analyzed = cached_analyze(
        df=df_raw,
        review_col=review_col,
        neutral_threshold=float(neutral_threshold),
        apply_preprocessing=bool(apply_preprocessing),
    )
    st.session_state["analysis_ready"] = True

# Apply search filter if provided
if search_query:
    mask = df_analyzed[review_col].astype(str).str.contains(search_query, case=False, na=False)
    df_view = df_analyzed[mask].copy()
else:
    df_view = df_analyzed.copy()

# Overview metrics
col_a, col_b, col_c = st.columns(3)
value_counts = df_view["sentiment"].value_counts(dropna=False)
col_a.metric("Total Reviews", int(len(df_view)))
col_b.metric("Positive", int(value_counts.get("Positive", 0)))
col_c.metric("Negative", int(value_counts.get("Negative", 0)))

# Distribution chart
st.subheader("Sentiment distribution")
counts_df = value_counts.rename_axis("sentiment").reset_index(name="count")
if chart_type == "Pie":
    fig = px.pie(counts_df, names="sentiment", values="count", color="sentiment", color_discrete_map={
        "Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"
    })
else:
    fig = px.bar(counts_df, x="sentiment", y="count", color="sentiment", color_discrete_map={
        "Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"
    }, text_auto=True)
    fig.update_layout(xaxis_title=None, yaxis_title="Count")

st.plotly_chart(fig, use_container_width=True)

# Example reviews per sentiment
st.subheader("Example reviews")
examples_container = st.container()
with examples_container:
    cols = st.columns(3)
    for idx, sentiment in enumerate(["Positive", "Negative", "Neutral"]):
        with cols[idx]:
            st.markdown(f"**{sentiment}**")
            subset = df_view[df_view["sentiment"] == sentiment]
            if subset.empty:
                st.caption("No examples available.")
                continue
            # Highest confidence first
            top_n = subset.sort_values("score", ascending=False).head(int(max_examples))
            for _, row in top_n.iterrows():
                st.write(f"- {row[review_col]}")

# Word clouds
st.subheader("Word clouds by sentiment")
if "processed_text" in df_analyzed.columns:
    wc_cols = st.columns(3)
    for idx, sentiment in enumerate(["Positive", "Negative", "Neutral"]):
        subset_texts = (
            df_view.loc[df_view["sentiment"] == sentiment, "processed_text"].dropna().astype(str).tolist()
        )
        with wc_cols[idx]:
            st.caption(sentiment)
            img = generate_wordcloud_image(subset_texts, max_words=int(max_wc_words))
            st.image(img, use_column_width=True)
else:
    st.caption("Word clouds unavailable (preprocessing disabled). Enable it in the sidebar.")

# Timeline (if timestamp provided)
if time_col_choice != "(none)":
    st.subheader("Timeline")
    parsed_time = try_parse_datetime(df_view, time_col_choice)
    if parsed_time.notna().any():
        tmp = df_view.copy()
        tmp["__time__"] = parsed_time.dt.floor("D")
        grouped = tmp.groupby(["__time__", "sentiment"]).size().reset_index(name="count")
        fig_t = px.line(
            grouped,
            x="__time__",
            y="count",
            color="sentiment",
            markers=True,
            color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"},
        )
        fig_t.update_layout(xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.caption("Could not parse the selected timestamp column.")

# Raw data expander
with st.expander("Show analyzed data"):
    st.dataframe(df_view, use_container_width=True)

st.caption(
    "Model: distilbert-base-uncased-finetuned-sst-2-english • Neutral if confidence < threshold"
)
