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

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 780px; }

.team-card {
    border-radius: 10px;
    border: 1px solid #333;
    padding: 1rem;
    text-align: center;
}
.team-name { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem; }
.stat-row { font-size: 0.85rem; color: #aaa; margin: 0.2rem 0; }
.stat-val { font-weight: 600; color: #eee; }

.verdict-box {
    border-radius: 12px;
    border: 1px solid #333;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.verdict-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
.verdict-sub { font-size: 0.9rem; color: #aaa; }

.prob-label {
    text-align: center;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 0.2rem;
}
.factor-item {
    border-left: 3px solid #444;
    padding: 0.5rem 0.75rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)


def format_season(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def verdict_text(home: str, away: str, home_p: float, draw_p: float, away_p: float) -> str:
    margin = abs(home_p - away_p)
    if home_p > away_p and home_p > draw_p:
        strength = "strong favorites" if margin > 0.20 else "slight favorites"
        return f"{home} are {strength} at home."
    elif away_p > home_p and away_p > draw_p:
        strength = "strong favorites" if margin > 0.20 else "slight favorites"
        return f"{away} are {strength} despite playing away."
    else:
        return "This matchup is too close to call. A draw is very much on the cards."


def contextual_explanation(feature: str, impact: float, home: str, away: str, snapshots: dict) -> str | None:
    """
    Turn a raw feature + coefficient impact into a plain-English sentence
    that includes actual stat values where available.
    """
    h = snapshots.get(home, {})
    a = snapshots.get(away, {})
    supports = impact > 0  # True = pushes toward the predicted outcome

    def val(snap, key, decimals=1):
        v = snap.get(key)
        return f"{v:.{decimals}f}" if v is not None else "?"

    lines = {
        "home_ppg_l5": (
            f"{home}'s recent form: {val(h,'ppg_l5')} pts/game over their last 5:"
            + ("this strengthens the home win case." if supports else "this weakens the home win case.")
        ),
        "away_ppg_l5": (
            f"{away}'s recent away form: {val(a,'ppg_l5')} pts/game over their last 5:"
            + ("their dip in form favors the home side." if supports else "their strong recent form makes an away win more likely.")
        ),
        "home_ppg_l10": (
            f"{home}'s longer-term form: {val(h,'ppg_l10')} pts/game over 10 games:"
            + ("consistent home strength." if supports else "inconsistency hurts their home win chances.")
        ),
        "away_ppg_l10": (
            f"{away}'s longer-term away record: {val(a,'ppg_l10')} pts/game over 10 games:"
            + ("their away struggles help the home side." if supports else "their strong away record closes the gap.")
        ),
        "home_gf_l5": (
            f"{home} have scored {val(h,'gf_l5')} goals/game recently:"
            + ("that firepower supports a home win." if supports else "a lack of goals makes a win harder.")
        ),
        "home_ga_l5": (
            f"{home} have conceded {val(h,'ga_l5')} goals/game recently:"
            + ("a tight defense helps." if supports else "a leaky defense increases the risk of dropping points.")
        ),
        "away_gf_l5": (
            f"{away} have scored {val(a,'gf_l5')} goals/game on the road:"
            + ("their limited attack reduces away win odds." if supports else "their threat in front of goal increases away win chances.")
        ),
        "away_ga_l5": (
            f"{away} have conceded {val(a,'ga_l5')} goals/game on the road:"
            + ("defensive fragility away from home helps {home}." if supports else f"a solid away defense makes life harder for {home}.")
        ),
        "home_rest_days": (
            f"{home} have had {int(h.get('rest_days', 0))} days rest:"
            + ("fresh legs help at home." if supports else "fatigue could be a factor.")
        ),
        "away_rest_days": (
            f"{away} have had {int(a.get('rest_days', 0))} days rest before travelling:"
            + ("travel fatigue may hurt them." if supports else "they arrive fresh, which favors an away result.")
        ),
        "ppg_diff_l5": (
            f"Recent form gap (last 5): {home} have {val(h,'ppg_l5')} pts/game vs {away}'s {val(a,'ppg_l5')}:"
            + ("the form advantage is with the home side." if supports else "the gap actually favors the visitors.")
        ),
        "gf_diff_l5": (
            f"Attacking output gap: {home} score {val(h,'gf_l5')} vs {away}'s {val(a,'gf_l5')} goals/game:"
            + ("home side has the sharper attack." if supports else "the away side's attack is more dangerous right now.")
        ),
    }
    return lines.get(feature)


# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    pipeline, snapshots, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root.")
    st.stop()

teams = sorted(snapshots.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Premier League Matchup Predictor")
st.markdown(
    "Pick any two Premier League teams and get a **pre-match win probability** "
    "based on each team's recent form:points per game, goals, and rest time. "
    "No score or result from the match itself is used."
)
st.divider()

# ── Team selectors ────────────────────────────────────────────────────────────
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

st.caption(f"Form stats through **{metadata['last_match_date']}**:the final match in the dataset.")
st.button("Predict", type="primary", use_container_width=True, key="predict_btn")

if st.session_state.get("predict_btn"):
    with st.spinner("Calculating..."):
        try:
            result = predict_matchup(home, away)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    probs = result["probabilities"]
    home_p  = probs["Home win"]
    draw_p  = probs["Draw"]
    away_p  = probs["Away win"]
    predicted = result["predicted"]

    h_snap = snapshots.get(home, {})
    a_snap = snapshots.get(away, {})

    st.divider()

    # ── Team form cards ───────────────────────────────────────────────────────
    st.markdown(f"### {home} vs {away}")
    st.caption("Current form heading into this fixture")

    def fmt(v, d=1):
        return f"{v:.{d}f}" if v is not None else "N/A"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='team-card'>"
            f"<div class='team-name'>{home} <span style='font-weight:400;font-size:0.85rem;color:#aaa;'>(Home)</span></div>"
            f"<div class='stat-row'>Points/game (last 5) <span class='stat-val'>{fmt(h_snap.get('ppg_l5'))}</span></div>"
            f"<div class='stat-row'>Points/game (last 10) <span class='stat-val'>{fmt(h_snap.get('ppg_l10'))}</span></div>"
            f"<div class='stat-row'>Goals scored/game <span class='stat-val'>{fmt(h_snap.get('gf_l5'))}</span></div>"
            f"<div class='stat-row'>Goals conceded/game <span class='stat-val'>{fmt(h_snap.get('ga_l5'))}</span></div>"
            f"<div class='stat-row'>Days since last match <span class='stat-val'>{int(h_snap.get('rest_days', 0))}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='team-card'>"
            f"<div class='team-name'>{away} <span style='font-weight:400;font-size:0.85rem;color:#aaa;'>(Away)</span></div>"
            f"<div class='stat-row'>Points/game (last 5) <span class='stat-val'>{fmt(a_snap.get('ppg_l5'))}</span></div>"
            f"<div class='stat-row'>Points/game (last 10) <span class='stat-val'>{fmt(a_snap.get('ppg_l10'))}</span></div>"
            f"<div class='stat-row'>Goals scored/game <span class='stat-val'>{fmt(a_snap.get('gf_l5'))}</span></div>"
            f"<div class='stat-row'>Goals conceded/game <span class='stat-val'>{fmt(a_snap.get('ga_l5'))}</span></div>"
            f"<div class='stat-row'>Days since last match <span class='stat-val'>{int(a_snap.get('rest_days', 0))}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability display ───────────────────────────────────────────────────
    st.markdown("#### Predicted outcome probabilities")
    st.caption(
        "A **win probability** is not a guarantee:it reflects how often a team with "
        "this form profile wins in similar matchups. 45% home win means: across many games "
        "with these stats, the home side wins about 45% of the time."
    )

    # Horizontal stacked bar
    bar_df = pd.DataFrame([
        {"Outcome": "Home win",  "Probability": home_p, "order": 0},
        {"Outcome": "Draw",      "Probability": draw_p, "order": 1},
        {"Outcome": "Away win",  "Probability": away_p, "order": 2},
    ])

    chart = (
        alt.Chart(bar_df)
        .mark_bar(height=44, cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Probability:Q", stack="normalize",
                    axis=alt.Axis(format=".0%", title=None, labels=False, ticks=False, domain=False)),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Home win", "Draw", "Away win"],
                    range=["#1e6f3a", "#8a6a00", "#1a3a6f"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=60)
    )
    st.altair_chart(chart, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{home_p:.0%}")
    c2.metric("Draw", f"{draw_p:.0%}")
    c3.metric(f"{away} win", f"{away_p:.0%}")

    st.info(verdict_text(home, away, home_p, draw_p, away_p))

    # ── Key factors ───────────────────────────────────────────────────────────
    st.markdown("#### What drove this prediction")
    st.caption(
        f"The model predicted **{predicted}**. Below are the five stats that had the "
        "most influence. Green means the stat pushed toward that outcome; "
        "red means it pushed against it."
    )

    for driver in result["top_drivers"]:
        explanation = contextual_explanation(
            driver["feature"], driver["impact"], home, away, snapshots
        )
        if not explanation:
            continue
        color = "#2ecc71" if driver["impact"] > 0 else "#e74c3c"
        direction = "supported" if driver["impact"] > 0 else "worked against"
        st.markdown(
            f"<div class='factor-item' style='border-left-color:{color};'>"
            f"{explanation}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Model details ─────────────────────────────────────────────────────────
    with st.expander("How the model works"):
        train_seasons = [format_season(s) for s in metadata["train_seasons"]]
        test_seasons  = [format_season(s) for s in metadata["test_seasons"]]
        m = metadata["metrics"]

        st.markdown(
            f"This is a **multinomial logistic regression**:a statistical model that "
            f"learns which combinations of form stats predict match outcomes. "
            f"It was trained on **{', '.join(train_seasons)}** Premier League seasons "
            f"and then tested on **{test_seasons[0]}**:a full season it had never seen."
        )
        st.markdown(
            "**Log loss** measures how well the model's probabilities match reality. "
            "Lower is better. A model that just always guesses the most common outcome "
            "scores around 1.05:this model scores 1.026, meaning its probabilities "
            "are better calibrated than that baseline."
        )

        perf_df = pd.DataFrame({
            "Model": [
                "Always predict home win",
                "Form rule (points threshold)",
                "This app (logistic regression)",
            ],
            "Log loss (lower is better)": [
                m["dummy_log_loss"],
                m["form_log_loss"],
                m["model_log_loss"],
            ],
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Accuracy on {test_seasons[0]}: **{m['model_accuracy']:.1%}** across 370 matches. "
            "Soccer is hard to predict:even professional betting markets are wrong ~45% of the time on hard games. "
            "Draws in particular are notoriously difficult; the model uses balanced class weighting "
            "to avoid always ignoring them."
        )
