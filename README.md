# NASCAR-USA

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

A NASCAR race-prediction pipeline that trains track-type-specific ensemble models and produces calibrated winner / top-3 / top-5 / top-10 probabilities for each driver in an upcoming race.

## Overview

NASCAR-USA turns historical race results and pre-race information into ranked, probabilistic race forecasts. It pulls upcoming race schedules, entry lists, and starting grids from the SportsRadar NASCAR API, engineers driver, team, track, and manufacturer features from a historical results store, and trains a separate model ensemble for each track type (short track, road course, superspeedway, intermediate). Predicted finishing positions are converted into win/top-N probabilities and calibrated with Platt scaling so the numbers can be compared meaningfully against bookmaker odds, which the project also scrapes.

The design goal is realism: features are built strictly from chronologically prior data to avoid leakage, the train/test split is time-based, and probabilities are calibrated rather than taken raw from the models.

## Key Features

- **Track-type-specific ensembles** — a fresh 4-model ensemble (XGBoost, LightGBM, RandomForest, GradientBoosting) is trained per track type, with a minimum-sample guard before training.
- **Dedicated top-5 models** — per-track-type model selection (deep LightGBM, XGBoost, or RandomForest) tuned for the top-5 target.
- **Position blending** — model-predicted positions are blended with the actual starting grid using per-track-type ratios before probabilities are derived.
- **Leak-free feature engineering** — recency-weighted driver form (last 3/5/10), track-specific history, team momentum, manufacturer-by-track-type averages, and win rate by track type, all computed from prior races only.
- **Probability calibration** — Platt scaling (logistic regression over raw probabilities) for winner, top-3, top-5, and top-10 targets.
- **Absolute and relative probabilities** — two scoring methods, each evaluated with a precision-at-N report broken down by track type.
- **Odds scraping** — Selenium/BeautifulSoup scrapers collect sportsbook winner odds for market comparison.
- **Model persistence** — trained ensembles, top-5 models, calibrators, encoders, medians, feature list, and config are pickled, alongside a generated `feature_requirements.json` describing the exact inputs a prediction needs.

## How It Works

1. **Fetch** — `a_fetch_upcoming.py` retrieves the upcoming race, entry list, and qualifying/grid data from SportsRadar and pulls historical context from PostgreSQL.
2. **Engineer features** — prior-race form, track history, team momentum, and manufacturer/track-type stats are computed chronologically.
3. **Split** — a time-based train/test split by race date prevents future data leaking into training.
4. **Train** — `a_model_generator.py` trains per-track-type ensembles and dedicated top-5 models.
5. **Predict & blend** — finishing positions are predicted per race (vectorized), blended with starting positions, and mapped to win/top-N probabilities.
6. **Calibrate & evaluate** — Platt scaling calibrates the probabilities; a precision report and sample race tables summarize quality.
7. **Persist** — all artifacts are written to a model directory for later inference.

## Tech Stack

- **Language:** Python 3
- **Modeling:** XGBoost, LightGBM, scikit-learn (RandomForest, GradientBoosting, LogisticRegression)
- **Data:** pandas, NumPy
- **Database:** PostgreSQL (historical results)
- **External API:** SportsRadar NASCAR API
- **Scraping:** Selenium, BeautifulSoup

## Getting Started

### Prerequisites

- Python 3.8+
- A SportsRadar NASCAR API key
- A PostgreSQL database containing historical NASCAR results
- A training dataset (`dataset_with_features.csv`) for `a_model_generator.py`

### Installation

```bash
git clone https://github.com/iampreetdave-max/NASCAR-USA.git
cd NASCAR-USA
pip install pandas numpy scikit-learn xgboost lightgbm psycopg2 selenium beautifulsoup4 requests
```

### Usage

Fetch the upcoming race and build features:

```bash
python a_fetch_upcoming.py
```

Train the ensembles and write model artifacts (reads `dataset_with_features.csv`, writes to `nascar_models/`):

```bash
python a_model_generator.py
```

Scrape sportsbook winner odds:

```bash
python scrape_winner_odds.py
```

## Configuration

API keys and database credentials are read by the fetch and scraping scripts. Provide your SportsRadar API key and PostgreSQL connection details as the scripts expect; do not commit real credentials to the repository.

## Project Structure

```
NASCAR-USA/
├── a_fetch_upcoming.py      # Fetch upcoming race data + build features (SportsRadar + PostgreSQL)
├── a_model_generator.py     # Train track-type ensembles, calibrate, persist models (v62)
├── data.py                  # Data processing utilities
├── fr.py                    # Feature engineering helpers
├── SCRAPE_ODDS.py           # Odds scraper
├── scrape_winner_odds.py    # Winner odds scraper
├── models/                  # Saved model artifacts
├── dataset.csv              # Historical race data
├── LICENSE
└── README.md
```

## License

Licensed under the [Apache License 2.0](LICENSE).
