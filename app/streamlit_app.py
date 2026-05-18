import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_matchup  # noqa: E402

st.set_page_config(page_title="Soccer Matchup Predictor", page_icon="⚽", layout="centered")

st.title("⚽ Premier League Matchup Predictor")
st.caption("Pre-match home / draw / away probabilities from recent form (no leakage).")

try:
    _, snapshots, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root first.")
    st.stop()

teams = sorted(snapshots.keys())
col1, col2 = st.columns(2)

with col1:
    home = st.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
with col2:
    away_options = [t for t in teams if t != home]
    away = st.selectbox("Away team", away_options, index=0)

st.info(f"Form stats through **{metadata['last_match_date']}** (last match in dataset).")

if st.button("Predict", type="primary"):
    if home == away:
        st.warning("Pick two different teams.")
    else:
        try:
            result = predict_matchup(home, away)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.subheader(f"{home} vs {away}")
        st.metric("Most likely outcome", result["predicted"])

        prob_df = pd.DataFrame(
            {"Outcome": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
        )
        st.bar_chart(prob_df.set_index("Outcome"))

        cols = st.columns(3)
        for col, (label, prob) in zip(cols, result["probabilities"].items()):
            col.metric(label, f"{prob:.0%}")

        st.markdown("**Why (top drivers for predicted class)**")
        for driver in result["top_drivers"]:
            direction = "favors" if driver["impact"] > 0 else "works against"
            st.write(f"- `{driver['feature']}` ({direction} prediction, impact {driver['impact']:+.3f})")

with st.expander("Model info"):
    st.write(
        f"Trained on seasons **{', '.join(metadata['train_seasons'])}**, "
        f"tested on **{', '.join(metadata['test_seasons'])}**."
    )
    st.json(metadata["metrics"])
