# Soccer Matchup Predictor

Pre-match **home / draw / away** probabilities for Premier League fixtures, built from rolling team form with strict temporal validation — no post-match data used at prediction time.

**[Live demo →](https://soccer-matchup-predictor.streamlit.app)**

## Highlights

- Multinomial logistic regression on 5 seasons of Premier League data ([football-data.co.uk](https://www.football-data.co.uk))
- **Time-based split**: train on 2019–20 → 2022–23, test unseen 2023–24 season
- Balanced class weighting so draws (the hardest outcome) are actually predicted
- Coefficient-based "why" panel — no black box

## Model performance (2023–24 hold-out)

| Model | Log loss ↓ | Notes |
|---|---|---|
| Majority class baseline | 1.054 | Always picks Home win |
| Form rule baseline | 1.041 | PPG threshold heuristic |
| **Logistic regression** | **1.026** | This app |

Accuracy: **50.3%** on 370 hold-out matches. Soccer draws lower accuracy relative to binary sports prediction — the calibrated probabilities matter more than hard labels.

## Features (pre-kickoff only)

| Feature | Description |
|---|---|
| `home/away_ppg_l5` | Points per game, last 5 matches |
| `home/away_ppg_l10` | Points per game, last 10 matches |
| `home/away_gf_l5` | Goals scored per game, last 5 |
| `home/away_ga_l5` | Goals conceded per game, last 5 |
| `home/away_rest_days` | Days since previous match |
| `ppg_diff_l5` | Form gap between teams |
| `gf_diff_l5` | Attack output gap |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train      # trains model, saves to models/
streamlit run app/streamlit_app.py
```

## Project layout

```
soccer-matchup-predictor/
  data/raw/             E0_1920.csv … E0_2324.csv (football-data.co.uk)
  src/
    load_data.py        Parse and concatenate season CSVs
    features.py         Rolling feature builder (no leakage)
    train.py            Train, evaluate, save model + snapshots
    predict.py          Load artifacts, build matchup features, predict
  models/
    model.joblib        Trained pipeline (StandardScaler + LogisticRegression)
    team_snapshots.joblib  Latest form stats per team
    metadata.json       Metrics, feature list, train/test seasons
  app/streamlit_app.py  Streamlit UI
  notebooks/01_eda.ipynb
```

## Leakage prevention

Features for match *M* use only matches with `Date < M.Date` for both teams. `FTHG`, `FTAG`, and `FTR` of the current row are never seen by the feature builder. Rolling windows require a minimum of 10 prior matches, so early-season cold-start rows are dropped rather than imputed.

## Deploy (Streamlit Community Cloud)

1. Push repo to GitHub (public)
2. [streamlit.io/cloud](https://streamlit.io/cloud) → New app → set main file: `app/streamlit_app.py`
3. No extra secrets or environment variables needed

## Data

Download Premier League (E0) files from [football-data.co.uk → England](https://www.football-data.co.uk/englandm.php) into `data/raw/` named `E0_1920.csv`, etc.

## Tech

Python · pandas · scikit-learn · Streamlit

## License

Data © football-data.co.uk. Code MIT.
