import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_data import load_matches  # noqa: E402
from src.predict import load_artifacts, predict_matchup  # noqa: E402

st.set_page_config(
    page_title="PL Matchup Predictor",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 820px; }

.team-card {
    border-radius: 10px;
    border: 1px solid #333;
    padding: 1.1rem 1rem;
}
.team-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
}
.team-name-label { font-size: 1.05rem; font-weight: 700; }
.role-label { font-size: 0.78rem; color: #888; }
.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    color: #aaa;
    padding: 0.18rem 0;
    border-bottom: 1px solid #2a2a2a;
}
.stat-row:last-child { border-bottom: none; }
.stat-val { font-weight: 600; color: #eee; }

.form-badge {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 4px;
    text-align: center;
    line-height: 22px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 3px;
}
.badge-W { background: #1e6f3a; color: #fff; }
.badge-D { background: #555; color: #fff; }
.badge-L { background: #8b1a1a; color: #fff; }

.factor-item {
    border-left: 3px solid #444;
    padding: 0.45rem 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    border-radius: 0 6px 6px 0;
    background: #1a1a1a;
}

.h2h-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.83rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid #222;
    color: #ccc;
}
.h2h-winner { font-weight: 700; color: #eee; }
.h2h-score { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ── Logo map (Premier League official badge CDN) ──────────────────────────────
PL_LOGO_IDS = {
    "Arsenal": 3, "Aston Villa": 7, "Bournemouth": 91, "Brentford": 94,
    "Brighton": 36, "Burnley": 90, "Chelsea": 8, "Crystal Palace": 31,
    "Everton": 11, "Fulham": 54, "Leeds": 2, "Leicester": 13,
    "Liverpool": 14, "Luton": 102, "Man City": 43, "Man United": 1,
    "Newcastle": 4, "Norwich": 45, "Nott'm Forest": 17, "Sheffield United": 49,
    "Southampton": 20, "Tottenham": 6, "Watford": 57, "West Brom": 35,
    "West Ham": 21, "Wolves": 39,
}

def logo_url(team: str) -> str | None:
    tid = PL_LOGO_IDS.get(team)
    return f"https://resources.premierleague.com/premierleague/badges/rb/t{tid}.svg" if tid else None


def format_season(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def form_badges_html(form: list[str]) -> str:
    badges = "".join(
        f"<span class='form-badge badge-{r}'>{r}</span>" for r in form
    )
    return f"<div style='margin-top:0.5rem;'>{badges}</div>"


def verdict_text(home: str, away: str, home_p: float, draw_p: float, away_p: float) -> str:
    margin = abs(home_p - away_p)
    if home_p > away_p and home_p > draw_p:
        strength = "strong favorites" if margin > 0.20 else "slight favorites"
        return f"{home} are {strength} at home."
    elif away_p > home_p and away_p > draw_p:
        strength = "strong favorites" if margin > 0.20 else "slight favorites"
        return f"{away} are {strength} despite playing away."
    else:
        return "This matchup is very close. A draw is firmly on the cards."


def contextual_explanation(feature: str, impact: float, home: str, away: str, snapshots: dict) -> str | None:
    h = snapshots.get(home, {})
    a = snapshots.get(away, {})
    supports = impact > 0

    def val(snap, key, d=1):
        v = snap.get(key)
        return f"{v:.{d}f}" if v is not None else "?"

    lines = {
        "home_ppg_l5": (
            f"{home} are averaging {val(h,'ppg_l5')} points per game over their last 5 matches. "
            + ("That consistent home form supports a win here." if supports else "That form doesn't inspire much confidence at home.")
        ),
        "away_ppg_l5": (
            f"{away} are averaging {val(a,'ppg_l5')} points per game over their last 5. "
            + ("Their away form has dipped, which favors the home side." if supports else "That strong recent form makes them a real threat on the road.")
        ),
        "home_ppg_l10": (
            f"{home} have averaged {val(h,'ppg_l10')} points per game over their last 10 matches. "
            + ("Solid longer-term form backs up a home win." if supports else "Their inconsistency over the last 10 games is a concern.")
        ),
        "away_ppg_l10": (
            f"{away} have averaged {val(a,'ppg_l10')} points per game over their last 10. "
            + ("Their away struggles over that stretch help the home side." if supports else "That strong long-term record closes the gap significantly.")
        ),
        "home_gf_l5": (
            f"{home} have scored {val(h,'gf_l5')} goals per game over their last 5. "
            + ("A sharp attack increases their chances at home." if supports else "Misfiring in front of goal makes a home win harder to come by.")
        ),
        "home_ga_l5": (
            f"{home} have conceded {val(h,'ga_l5')} goals per game recently. "
            + ("A tight defense at home is a big advantage." if supports else "Conceding freely at home is a problem against quality opposition.")
        ),
        "away_gf_l5": (
            f"{away} have scored {val(a,'gf_l5')} goals per game on the road recently. "
            + ("A muted attack away from home reduces their threat." if supports else "Scoring freely on the road makes them dangerous.")
        ),
        "away_ga_l5": (
            f"{away} have conceded {val(a,'ga_l5')} goals per game on the road. "
            + (f"Defensive fragility away from home gives {home} an opening." if supports else f"A solid away defense will make it difficult for {home}.")
        ),
        "home_rest_days": (
            f"{home} have had {int(h.get('rest_days', 0))} days since their last match. "
            + ("Fresh legs at home is an advantage." if supports else "Fatigue could be a factor for the home side.")
        ),
        "away_rest_days": (
            f"{away} have had {int(a.get('rest_days', 0))} days since their last match before travelling. "
            + ("Less recovery time on the road is a disadvantage for the visitors." if supports else "They arrive well-rested, which helps away sides.")
        ),
        "ppg_diff_l5": (
            f"The recent form gap: {home} at {val(h,'ppg_l5')} vs {away} at {val(a,'ppg_l5')} points per game (last 5). "
            + ("The form advantage is clearly with the home side." if supports else "The gap actually favors the visitors right now.")
        ),
        "gf_diff_l5": (
            f"Attacking output: {home} scoring {val(h,'gf_l5')} vs {away} scoring {val(a,'gf_l5')} goals per game (last 5). "
            + ("The home side has the sharper attack." if supports else "The away side's attack has been more potent recently.")
        ),
    }
    return lines.get(feature)


@st.cache_data(show_spinner=False)
def get_h2h(home: str, away: str) -> pd.DataFrame:
    matches = load_matches()
    mask = (
        ((matches["HomeTeam"] == home) & (matches["AwayTeam"] == away)) |
        ((matches["HomeTeam"] == away) & (matches["AwayTeam"] == home))
    )
    return matches[mask].sort_values("Date", ascending=False).head(8).reset_index(drop=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    pipeline, snapshots, metadata = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root.")
    st.stop()

teams = sorted(snapshots.keys())

# ── Shareable URL: read query params ─────────────────────────────────────────
params = st.query_params
url_home = params.get("home", "")
url_away = params.get("away", "")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Premier League Matchup Predictor")
st.markdown(
    "Pick any two Premier League teams and get a pre-match win probability "
    "based on each team's recent form: points per game, goals, and rest time. "
    "No result or score from the match itself is used."
)
st.divider()

# ── Team selectors + flip button ─────────────────────────────────────────────
if "home" not in st.session_state:
    st.session_state.home = url_home if url_home in teams else (teams.index("Arsenal") and "Arsenal")
if "away" not in st.session_state:
    away_default = url_away if url_away in teams and url_away != st.session_state.home else "Chelsea"
    st.session_state.away = away_default

col1, mid, col2 = st.columns([5, 1, 5])
with col1:
    st.markdown("**Home team**")
    home = st.selectbox(
        "Home team", teams,
        index=teams.index(st.session_state.home) if st.session_state.home in teams else 0,
        label_visibility="collapsed", key="home_select",
    )
with mid:
    st.markdown("<div style='padding-top:1.7rem;text-align:center;'>", unsafe_allow_html=True)
    if st.button("swap", use_container_width=True, help="Swap home and away teams"):
        st.session_state.home, st.session_state.away = st.session_state.away, st.session_state.home
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("**Away team**")
    away_options = [t for t in teams if t != home]
    away_idx = away_options.index(st.session_state.away) if st.session_state.away in away_options else 0
    away = st.selectbox(
        "Away team", away_options, index=away_idx,
        label_visibility="collapsed", key="away_select",
    )

st.session_state.home = home
st.session_state.away = away

st.caption(f"Form data through **{metadata['last_match_date']}**")
st.button("Predict", type="primary", use_container_width=True, key="predict_btn")

if st.session_state.get("predict_btn"):
    # Write teams to URL for shareability
    st.query_params["home"] = home
    st.query_params["away"] = away

    with st.spinner("Calculating..."):
        try:
            result = predict_matchup(home, away)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    probs   = result["probabilities"]
    home_p  = probs["Home win"]
    draw_p  = probs["Draw"]
    away_p  = probs["Away win"]
    predicted = result["predicted"]

    h_snap = snapshots.get(home, {})
    a_snap = snapshots.get(away, {})

    st.divider()

    # ── Team cards ────────────────────────────────────────────────────────────
    def team_card_html(team: str, role: str, snap: dict) -> str:
        logo = logo_url(team)
        logo_html = (
            f"<img src='{logo}' style='height:36px;width:36px;object-fit:contain;' />"
            if logo else ""
        )
        form = snap.get("recent_form", [])
        badges = form_badges_html(form) if form else ""

        def r(key, d=1):
            v = snap.get(key)
            return f"{v:.{d}f}" if v is not None else "N/A"

        return (
            f"<div class='team-card'>"
            f"<div class='team-header'>"
            f"{logo_html}"
            f"<div><div class='team-name-label'>{team}</div>"
            f"<div class='role-label'>{role}</div></div>"
            f"</div>"
            f"{badges}"
            f"<div style='margin-top:0.6rem;'>"
            f"<div class='stat-row'><span>Points/game (last 5)</span><span class='stat-val'>{r('ppg_l5')}</span></div>"
            f"<div class='stat-row'><span>Points/game (last 10)</span><span class='stat-val'>{r('ppg_l10')}</span></div>"
            f"<div class='stat-row'><span>Goals scored/game</span><span class='stat-val'>{r('gf_l5')}</span></div>"
            f"<div class='stat-row'><span>Goals conceded/game</span><span class='stat-val'>{r('ga_l5')}</span></div>"
            f"<div class='stat-row'><span>Days since last match</span><span class='stat-val'>{int(snap.get('rest_days', 0))}</span></div>"
            f"</div>"
            f"</div>"
        )

    c1, c2 = st.columns(2)
    c1.markdown(team_card_html(home, "Home", h_snap), unsafe_allow_html=True)
    c2.markdown(team_card_html(away, "Away", a_snap), unsafe_allow_html=True)

    st.caption(
        "Form badges show the last 5 results (oldest to newest): "
        "W = win, D = draw, L = loss. Points/game uses 3 for a win, 1 for a draw, 0 for a loss."
    )

    # ── Upset alert ───────────────────────────────────────────────────────────
    if away_p > home_p + 0.05:
        st.warning(
            f"Upset alert: {away} are favored despite playing away from home. "
            "Away wins are rarer in the Premier League, making this a notable prediction."
        )
    elif abs(home_p - away_p) <= 0.05:
        st.info("Coin flip: the model sees very little to separate these teams. Any result is on the table.")

    # ── Probabilities ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Predicted outcome probabilities")
    st.caption(
        "A win probability is not a guarantee. It reflects how often a team in this form situation "
        "wins similar matchups. A 45% home win probability means the home side wins roughly 45 times "
        "out of 100 games with these form numbers going in."
    )

    bar_df = pd.DataFrame([
        {"Outcome": f"{home} win", "Probability": home_p, "order": 0},
        {"Outcome": "Draw",        "Probability": draw_p, "order": 1},
        {"Outcome": f"{away} win", "Probability": away_p, "order": 2},
    ])

    chart = (
        alt.Chart(bar_df)
        .mark_bar(height=48, cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Probability:Q", stack="normalize",
                    axis=alt.Axis(format=".0%", title=None, labels=False, ticks=False, domain=False)),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=[f"{home} win", "Draw", f"{away} win"],
                    range=["#1e6f3a", "#7a6000", "#1a3a6f"],
                ),
                legend=alt.Legend(orient="bottom", title=None, labelFontSize=12),
            ),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=64)
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
        f"The model predicted **{predicted}**. These are the five stats that had the most influence. "
        "A green bar means the stat pushed the model toward that outcome; "
        "red means it pushed against it."
    )

    for driver in result["top_drivers"]:
        explanation = contextual_explanation(
            driver["feature"], driver["impact"], home, away, snapshots
        )
        if not explanation:
            continue
        color = "#2ecc71" if driver["impact"] > 0 else "#e74c3c"
        st.markdown(
            f"<div class='factor-item' style='border-left-color:{color};'>{explanation}</div>",
            unsafe_allow_html=True,
        )

    # ── Head-to-head ──────────────────────────────────────────────────────────
    st.markdown("#### Head-to-head history")
    st.caption("Last 8 meetings between these clubs across all seasons in the dataset.")

    h2h = get_h2h(home, away)
    if h2h.empty:
        st.write("No meetings found between these clubs in the dataset.")
    else:
        home_wins = sum(
            1 for _, r in h2h.iterrows()
            if (r["HomeTeam"] == home and r["FTR"] == "H") or (r["HomeTeam"] == away and r["FTR"] == "A")
        )
        away_wins = sum(
            1 for _, r in h2h.iterrows()
            if (r["HomeTeam"] == away and r["FTR"] == "H") or (r["HomeTeam"] == home and r["FTR"] == "A")
        )
        draws = len(h2h) - home_wins - away_wins

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(f"{home} wins", home_wins)
        sc2.metric("Draws", draws)
        sc3.metric(f"{away} wins", away_wins)

        rows_html = ""
        for _, r in h2h.iterrows():
            ht, at = r["HomeTeam"], r["AwayTeam"]
            hg, ag = int(r["FTHG"]), int(r["FTAG"])
            ftr = r["FTR"]
            winner = ht if ftr == "H" else (at if ftr == "A" else None)
            date_str = r["Date"].strftime("%d %b %Y")

            def bold(team):
                return f"<span class='h2h-winner'>{team}</span>" if team == winner else team

            rows_html += (
                f"<div class='h2h-row'>"
                f"<span>{date_str}</span>"
                f"<span>{bold(ht)} <span class='h2h-score'>{hg} - {ag}</span> {bold(at)}</span>"
                f"<span style='color:#888;font-size:0.78rem;'>{'Draw' if not winner else winner + ' win'}</span>"
                f"</div>"
            )
        st.markdown(f"<div style='margin-top:0.5rem;'>{rows_html}</div>", unsafe_allow_html=True)

    # ── Share link ────────────────────────────────────────────────────────────
    share_url = f"https://whowillwin.streamlit.app/?home={home.replace(' ', '+')}&away={away.replace(' ', '+')}"
    st.markdown("<br>", unsafe_allow_html=True)
    st.code(share_url, language=None)
    st.caption("Copy the link above to share this exact matchup.")

    # ── Model details ─────────────────────────────────────────────────────────
    with st.expander("How the model works"):
        train_seasons = [format_season(s) for s in metadata["train_seasons"]]
        test_seasons  = [format_season(s) for s in metadata["test_seasons"]]
        m = metadata["metrics"]

        st.markdown(
            f"This is a **multinomial logistic regression**, a statistical model that learns which "
            f"combinations of form stats historically predict match outcomes. "
            f"It was trained on **{', '.join(train_seasons)}** Premier League seasons "
            f"and evaluated on **{test_seasons[0]}**, a full season it never saw during training."
        )
        st.markdown(
            "**Log loss** measures how well-calibrated the model's probabilities are. "
            "Lower is better. A model that always guesses the most common outcome scores around 1.08. "
            "This model scores 1.04, meaning its probability estimates are more accurate than that."
        )
        perf_df = pd.DataFrame({
            "Model": [
                "Always predict home win",
                "Form rule (points threshold)",
                "This app (logistic regression)",
            ],
            "Log loss (lower is better)": [
                m["dummy_log_loss"], m["form_log_loss"], m["model_log_loss"],
            ],
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Accuracy on {test_seasons[0]}: **{m['model_accuracy']:.1%}** across "
            f"{int(sum(metadata['confusion_matrix'][i][j] for i in range(3) for j in range(3)))} matches. "
            "Soccer is inherently hard to predict. Draws are the toughest outcome; "
            "the model uses balanced class weighting to avoid always ignoring them."
        )
