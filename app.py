# -*- coding: utf-8 -*-
"""Fake News Detection — Streamlit UI (Bidirectional LSTM only)
"""

import os
import re

import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D

# ── Config ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = "saved_models"
MAX_VOCAB = 30_000
MAX_LEN = 300
EMBED_DIM = 64
MODEL_NAME = "Bidirectional_LSTM"

# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning (must match training preprocessing exactly)
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = ['Reuters', 'AP', 'NYT', 'CNN', 'BBC', 'AFP', 'Non-US', 'Media', 'Verified']


def _strip_source_prefix(text: str) -> str:
    """Strip source attribution prefix from the start of an article."""
    text = str(text)
    # Try patterns like "WASHINGTON (Reuters) - " or "SEATTLE/WASHINGTON (Reuters) - "
    for source in SOURCES:
        # Pattern: location/source markers before the dash
        pattern = re.compile(
            r'^'                                      # start of string
            r'[A-Za-z\s/]+?'                          # location (non-greedy)
            r'\s*\(' + source + r'\)\s*-\s*'           # (Source) - with optional spaces
            , re.IGNORECASE
        )
        text = pattern.sub('', text, count=1)
        # Also try bare "(Source) - "
        pattern2 = re.compile(
            r'^\(' + source + r'\)\s*-\s*'
            , re.IGNORECASE
        )
        text = pattern2.sub('', text, count=1)
    return text


def _clean(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = _strip_source_prefix(text)  # strip source prefix (matches training)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake News Detector")
st.markdown(
    "Paste a news article below. The **title** is optional — "
    "the **body text** is required. The Bidirectional LSTM model will "
    "classify it as **REAL** or **FAKE**."
)
st.divider()

# ── Input section ──────────────────────────────────────────────────────────────
st.subheader("📝 Article Input")

col_title, col_text = st.columns([1, 3])

with col_title:
    title = st.text_input("Headline (optional)", placeholder="Breaking News…")

with col_text:
    text = st.text_area(
        "Article body (required)",
        placeholder="Paste the full article text here…",
        height=220,
    )

submitted = st.button("🔎  Detect Fake News", type="primary", use_container_width=True)

st.divider()

# ── Results ────────────────────────────────────────────────────────────────────
st.subheader("📊 Prediction Result")

if submitted:
    if not text.strip():
        st.warning("Please enter some article text before detecting.")
    else:
        with st.spinner("Loading model…"):
            # Load tokenizer
            tokenizer_path = os.path.join(WEIGHTS_DIR, "tokenizer.json")
            with open(tokenizer_path, encoding="utf-8") as f:
                tokenizer = tokenizer_from_json(f.read())
            vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

            # Build model architecture
            model = Sequential([
                Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
                SpatialDropout1D(0.2),
                Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
                Dense(32, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid")
            ], name="Bidirectional_LSTM")

            model.compile(optimizer="adam", loss="binary_crossentropy")

            # Load weights
            weights_path = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}.weights.h5")
            _ = model.predict(np.zeros((1, MAX_LEN), dtype="int32"), verbose=0)
            model.load_weights(weights_path)

        with st.spinner("Running inference…"):
            combined = _clean(f"{title} {text}" if title else text)
            seq = tokenizer.texts_to_sequences([combined])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

            prob = float(model.predict(padded, verbose=0)[0][0])
            label = "REAL" if prob >= 0.5 else "FAKE"
            confidence = prob if prob >= 0.5 else 1 - prob
            prob_fake = 1 - prob

        # Free memory
        del model
        tf.keras.backend.clear_session()

        # Display verdict
        if label == "REAL":
            st.markdown(
                "## 🟢 VERDICT: **REAL** NEWS"
            )
            bg = "background-color:#1a3d2e; border:1px solid #2ecc71;"
        else:
            st.markdown(
                "## 🔴 VERDICT: **FAKE** NEWS"
            )
            bg = "background-color:#3d1a1a; border:1px solid #e74c3c;"

        bar_len = 30
        real_bar = "█" * int(prob * bar_len) + "░" * (bar_len - int(prob * bar_len))
        fake_bar = "█" * int(prob_fake * bar_len) + "░" * (bar_len - int(prob_fake * bar_len))

        st.markdown(
            f"""
            <div style="border-radius:12px; padding:24px; {bg}">
                <div style="font-size:1.1em; margin-bottom:12px;">
                    Confidence: <b>{confidence:.1%}</b>
                </div>
                <div style="margin-bottom:6px;">
                    <b>P(real)</b> &nbsp;[{real_bar}]&nbsp; {prob:.2%}
                </div>
                <div>
                    <b>P(fake)</b> &nbsp;[{fake_bar}]&nbsp; {prob_fake:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<small style='color:#888;'>Model: Bidirectional LSTM &nbsp;|&nbsp; "
            f"Threshold: 0.5 &nbsp;|&nbsp; MAX_LEN: {MAX_LEN}</small>",
            unsafe_allow_html=True,
        )

else:
    st.info("Enter an article above and click **Detect Fake News** to start.")

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("ℹ️  How this works"):
    st.markdown(
        "1. **Input** — Your article title (optional) and body are combined, "
        "cleaned (lowercased, URLs/HTML removed), and tokenized.\n"
        "2. **Padding** — Text is padded/truncated to 300 tokens (MAX_LEN).\n"
        "3. **Inference** — The Bidirectional LSTM produces a probability P(real).\n"
        "4. **Verdict** — P(real) ≥ 0.5 → **REAL**, else → **FAKE**.\n"
        "5. **Confidence** — The probability assigned to the predicted class."
    )
