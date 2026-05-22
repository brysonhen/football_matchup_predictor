import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_matchup  # noqa: E402

st.set_page_config(
    page_title="PL Matchup Predictor",
    page_icon="",
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
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_LABELS = {
    "home_ppg_l5":  "Home points/game — last 5",
    "away_ppg_l5":  "Away points/game — last 5",
    "home_ppg_l10": "Home points/game — last 10",
    "away_ppg_l10": "Away points/game — last 10",
    "home_gf_l5":   "Home goals scored/game — last 5",
    "home_ga_l5":   "Home goals conceded/game — last 5",
    "away_gf_l5":   "Away goals scored/game — last 5",
    "away_ga_l5":   "Away goals conceded/game — last 5",
    "home_rest_days": "Home days since last match",
    "away_rest_days": "Away days since last match",
    "ppg_diff_l5":  "Form gap (home minus away points/game, last 5)",
    "gf_diff_l5":   "Attack gap (home minus away goals/game, last 5)",
}

OUTCOME_COLORS = {
    "Home win": "#1e6f3a",
    "Draw":     "#9b6a00",
    "Away win": "#1a3a6f",
}


def format_season(code: str) -> str:
    """'1920' -> '2019-20', '2324' -> '2023-24'"""
    return f"20{code[:2]}-{code[2:]}"


try:
    pipeline, snapshots, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root first.")
    st.stop()

st.title("Premier League Matchup Predictor")
st.caption("Pre-match win probabilities based on rolling team form. No post-match data is used.")

teams = sorted(snapshots.keys())

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Home team**")
    default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    home = st.selectbox("Home team", teams, index=default_home, label_visibility="collapsed")

with col2:
    st.markdown("**Away team**")
    away_options = [t for t in teams if t != home]
    default_away = away_options.index("Chelsea") if "Chelsea" in away_options else 0
    away = st.selectbox("Away team", away_options, index=default_away, label_visibility="collapsed")

st.caption(f"Form data through **{metadata['last_match_date']}** — the last match in the dataset.")

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

    c1, c2, c3 = st.columns(3)
    for col, label, prob in [(c1, "Home win", home_p), (c2, "Draw", draw_p), (c3, "Away win", away_p)]:
        color = OUTCOME_COLORS[label]
        if label == predicted:
            border = f"3px solid {color}"
            tag = "<div style='font-size:0.75rem; margin-top:4px;'>predicted</div>"
        else:
            border = "1px solid #444"
            tag = ""
        col.markdown(
            f"<div class='outcome-box' style='border:{border}; color:{color};'>"
            f"<div style='font-size:1.6rem;'>{prob:.0%}</div>"
            f"<div style='font-size:0.85rem; margin-top:4px;'>{label}</div>"
            f"{tag}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        "The bars below show the model's confidence in each outcome. "
        "Probabilities sum to 100%."
    )

    bar_df = pd.DataFrame({
        "Outcome": ["Home win", "Draw", "Away win"],
        "Probability": [home_p, draw_p, away_p],
        "Color": [OUTCOME_COLORS["Home win"], OUTCOME_COLORS["Draw"], OUTCOME_COLORS["Away win"]],
    })

    chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Outcome:N",
                sort=["Home win", "Draw", "Away win"],
                axis=alt.Axis(labelAngle=0, title=None),
            ),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="Probability"),
            ),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Home win", "Draw", "Away win"],
                    range=[OUTCOME_COLORS["Home win"], OUTCOME_COLORS["Draw"], OUTCOME_COLORS["Away win"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**What drove this prediction**")
    st.caption(
        f"The model predicted **{predicted}**. Each factor below shows how much it pushed "
        "the model toward or away from that outcome. Positive means it supported the prediction; "
        "negative means it worked against it."
    )
    for driver in result["top_drivers"]:
        raw = driver["feature"]
        label = FEATURE_LABELS.get(raw, raw.replace("_", " "))
        impact = driver["impact"]
        direction = "supported" if impact > 0 else "worked against"
        color = "#2ecc71" if impact > 0 else "#e74c3c"
        sign = "+" if impact > 0 else ""
        st.markdown(
            f"<div style='padding:0.3rem 0; font-size:0.9rem;'>"
            f"<b>{label}</b><br>"
            f"<span style='color:{color};'>{direction} the prediction</span>"
            f" &nbsp; <code>{sign}{impact:.3f}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

with st.expander("Model details"):
    m = metadata["metrics"]
    train_seasons = [format_season(s) for s in metadata["train_seasons"]]
    test_seasons  = [format_season(s) for s in metadata["test_seasons"]]

    st.markdown(
        f"Trained on **{', '.join(train_seasons)}** seasons, evaluated on the "
        f"**{', '.join(test_seasons)}** season (data the model never saw during training)."
    )
    st.markdown(
        "Log loss measures how well the model's probabilities match real outcomes — lower is better. "
        "A perfectly random guess scores around 1.10 for a three-outcome sport."
    )

    perf_df = pd.DataFrame({
        "Model": [
            "Always predict home win",
            "Form rule (PPG threshold)",
            "Logistic regression (this app)",
        ],
        "Log loss (lower is better)": [
            m["dummy_log_loss"],
            m["form_log_loss"],
            m["model_log_loss"],
        ],
    })
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    st.caption(
        f"Accuracy on {test_seasons[0]} hold-out: **{m['model_accuracy']:.1%}** across 370 matches. "
        "Draws are the hardest outcome to predict in soccer — the model uses balanced class weighting "
        "so draws are actually considered rather than always ignored."
    )
