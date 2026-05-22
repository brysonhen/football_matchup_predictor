import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_matchup  # noqa: E402

st.set_page_config(
    page_title="PL Matchup Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .outcome-box {
        border-radius: 12px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        font-weight: 600;
    }
    .driver-row { padding: 0.25rem 0; font-size: 0.9rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Premier League Matchup Predictor")
st.caption(
    "Pre-match win probabilities from rolling form — no post-match data used."
)

try:
    pipeline, snapshots, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root first.")
    st.stop()

teams = sorted(snapshots.keys())

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🏠 Home team**")
    default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    home = st.selectbox("Home team", teams, index=default_home, label_visibility="collapsed")

with col2:
    st.markdown("**✈️ Away team**")
    away_options = [t for t in teams if t != home]
    default_away = away_options.index("Chelsea") if "Chelsea" in away_options else 0
    away = st.selectbox("Away team", away_options, index=default_away, label_visibility="collapsed")

st.caption(f"Form stats through **{metadata['last_match_date']}** (last match in dataset).")

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    with st.spinner("Calculating..."):
        try:
            result = predict_matchup(home, away)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    st.divider()
    st.subheader(f"{home}  vs  {away}")

    probs = result["probabilities"]
    home_p = probs["Home win"]
    draw_p = probs["Draw"]
    away_p = probs["Away win"]
    predicted = result["predicted"]

    def _color(label: str) -> str:
        return {
            "Home win": "#1e6f3a",
            "Draw":     "#9b6a00",
            "Away win": "#1a3a6f",
        }[label]

    c1, c2, c3 = st.columns(3)
    for col, label, prob in [
        (c1, "Home win", home_p),
        (c2, "Draw", draw_p),
        (c3, "Away win", away_p),
    ]:
        border = "3px solid" if label == predicted else "1px solid #dee2e6"
        col.markdown(
            f"""<div class="outcome-box" style="border:{border} {_color(label)}; color:{_color(label)};">
            <div style="font-size:1.6rem;">{prob:.0%}</div>
            <div style="font-size:0.85rem; margin-top:4px;">{label}</div>
            {"<div style='font-size:0.75rem;'>★ predicted</div>" if label == predicted else ""}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    bar_data = pd.DataFrame(
        {"Probability": [home_p, draw_p, away_p]},
        index=["Home win", "Draw", "Away win"],
    )
    st.bar_chart(bar_data, color=["#1e6f3a"], height=200)

    st.markdown("**Top prediction drivers**")
    for driver in result["top_drivers"]:
        direction = "↑ favors" if driver["impact"] > 0 else "↓ works against"
        feature_label = driver["feature"].replace("_", " ")
        impact_str = f"{driver['impact']:+.3f}"
        st.markdown(
            f"<div class='driver-row'>• <b>{feature_label}</b> &nbsp; "
            f"<span style='color:{'#1e6f3a' if driver['impact']>0 else '#c0392b'}'>"
            f"{direction} prediction</span> ({impact_str})</div>",
            unsafe_allow_html=True,
        )

with st.expander("Model performance"):
    m = metadata["metrics"]
    perf_df = pd.DataFrame(
        {
            "Model": ["Majority class baseline", "Form rule baseline", "Logistic regression (this app)"],
            "Log loss ↓": [m["dummy_log_loss"], m["form_log_loss"], m["model_log_loss"]],
        }
    )
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    st.caption(
        f"Accuracy on 2023–24 hold-out: **{m['model_accuracy']:.1%}**. "
        "Train seasons: 2019–20 → 2022–23. Draw recall is intentionally non-zero "
        "via balanced class weighting — soccer draws are notoriously hard to predict."
    )
    st.write(
        f"Trained on seasons **{', '.join(metadata['train_seasons'])}**, "
        f"tested on **{', '.join(metadata['test_seasons'])}**."
    )
