# NASCAR-USA

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C35?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)

> A NASCAR race prediction pipeline using ensemble ML models with track-type-specific training, odds scraping, and probability calibration.

## About

A comprehensive NASCAR prediction system that fetches upcoming race data from the SportsRadar API, retrieves historical performance from a PostgreSQL database, engineers driver/team/track features, trains track-type-specific ensemble models (XGBoost, LightGBM, RandomForest, GradientBoosting), and generates calibrated winner/top-3/top-5/top-10 probabilities. Includes odds scraping for market comparison and validation.

## Tech Stack

- **Language:** Python 3
- **ML:** XGBoost, LightGBM, RandomForest, GradientBoosting
- **Calibration:** Platt scaling (sklearn CalibratedClassifierCV)
- **Data:** Pandas, NumPy, scikit-learn
- **Database:** PostgreSQL
- **API:** SportsRadar NASCAR API
- **Scraping:** Selenium, BeautifulSoup

## Features

- **Track-type-specific models** — separate ensembles trained for each track type (oval, road course, superspeedway)
- **Rich feature engineering** — driver form, track-specific history, team momentum, manufacturer averages, qualifying data
- **Ensemble predictions** — weighted combination of XGBoost, LightGBM, RF, and GB
- **Probability calibration** — Platt scaling for well-calibrated win/top-N probabilities
- **Odds scraping** — fetches market odds from sportsbook websites
- **Historical database** — PostgreSQL-backed storage of past race results
- **SportsRadar integration** — live race schedules, entry lists, qualifying, and starting grids
- **Model persistence** — trained models saved as PKL files

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database with historical NASCAR data
- SportsRadar API key

### Installation

```bash
git clone https://github.com/iampreetdave-max/NASCAR-USA.git
cd NASCAR-USA
pip install pandas numpy scikit-learn xgboost lightgbm psycopg2 selenium beautifulsoup4 requests
```

### Run

**Fetch upcoming race data and generate features:**

```bash
python a_fetch_upcoming.py
```

**Train models:**

```bash
python a_model_generator.py
```

**Scrape odds:**

```bash
python scrape_winner_odds.py
```

## Project Structure

```
NASCAR-USA/
├── a_fetch_upcoming.py      # Fetch races & generate features
├── a_model_generator.py     # Train ensemble models (v62)
├── data.py                  # Data processing utilities
├── fr.py                    # Feature engineering
├── SCRAPE_ODDS.py           # Odds scraper
├── scrape_winner_odds.py    # Winner odds scraper
├── models/                  # Trained model files (.pkl)
├── dataset.csv              # Historical race data
├── LICENSE
└── README.md
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
