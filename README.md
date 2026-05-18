# Soccer Matchup Predictor

Pre-match **home / draw / away** probabilities for Premier League matchups, built from rolling team form with strict temporal validation—no post-match leakage.

## Highlights

- Multinomial logistic regression on 5 seasons of [football-data.co.uk](https://www.football-data.co.uk) E0 CSVs
- Time-based split: train on 2019–20 → 2022–23, test on 2023–24
- Baselines: majority class + simple form rule vs ML (log loss)
- Streamlit app with probability bars and coefficient-based “why”

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (downloads not required if data/raw already has E0_*.csv)
python -m src.train

# Run app
streamlit run app/streamlit_app.py
```

## Project layout

```
soccer-matchup-predictor/
  data/raw/           # E0_1920.csv … E0_2324.csv
  src/                # load, features, train, predict
  models/             # model.joblib + metadata (generated)
  app/streamlit_app.py
  notebooks/01_eda.ipynb
```

## Data

Download Premier League (E0) files from [football-data.co.uk → England](https://www.football-data.co.uk/englandm.php) into `data/raw/` as `E0_1920.csv`, etc.

## Methodology

**Target:** `FTR` — H / D / A  

**Features (pre-kickoff only):** rolling points-per-game (5 & 10), goals for/against (5), rest days, form diffs  

**Evaluation:** log loss on hold-out season 2023–24  

See `models/metadata.json` after training for metrics and confusion matrix.

## Deploy

Push to GitHub, then [Streamlit Community Cloud](https://streamlit.io/cloud) → New app → `app/streamlit_app.py`.

## License

Data © football-data.co.uk. Code MIT.
