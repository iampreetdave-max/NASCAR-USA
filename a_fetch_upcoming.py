#!/usr/bin/env python3
"""
NASCAR UPCOMING RACE PREDICTION DATA FETCHER
Fetches data from SportsRadar API for upcoming races and generates features for prediction.
Falls back to database (nascar_usa_data) for historical features when needed.

USAGE:
    python fetch_upcoming_race.py

OUTPUT:
    upcoming_race_features.csv - Ready for model prediction
"""

import requests
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import time
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# SportsRadar API
API_KEY = "mr6vgIYkBwabuE5n210HD6zFxqn8VMVVIuEMGw3n"
BASE_URL = "https://api.sportradar.com/nascar-ot3"
SERIES = "mc"  # Monster Cup
SEASON = 2026

# Database Configuration (fallback for historical data)
DB_CONFIG = {
    'host': 'winbets-predictions.postgres.database.azure.com',
    'port': 5432,
    'database': 'postgres',
    'user': 'winbets',
    'password': 'Constantinople@1900',
    'sslmode': 'require'
}
DB_TABLE = 'nascar_usa_data'

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2

# =============================================================================
# API HELPER FUNCTIONS
# =============================================================================

def fetch_with_retry(url, max_retries=MAX_RETRIES):
    """Helper to retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"accept": "application/json"}, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print(f"  ERROR 403: API key may not have NASCAR subscription enabled")
                return None
            elif response.status_code == 404:
                print(f"  ERROR 404: Resource not found")
                return None
            else:
                print(f"  ERROR {response.status_code}: {response.text[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
        except Exception as e:
            print(f"  Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return None
    return None

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"  Database connection error: {e}")
        return None

# =============================================================================
# FETCH SCHEDULE & FIND UPCOMING RACE
# =============================================================================

def fetch_schedule():
    """Fetch 2026 NASCAR schedule"""
    print("\n" + "="*80)
    print(f"FETCHING {SEASON} NASCAR SCHEDULE")
    print("="*80)
    
    url = f"{BASE_URL}/{SERIES}/{SEASON}/races/schedule.json?api_key={API_KEY}"
    print(f"  URL: {url[:80]}...")
    
    data = fetch_with_retry(url)
    
    if not data:
        print("  ✗ Failed to fetch schedule")
        return None
    
    races = []
    for event in data.get("events", []):
        track = event.get("track", {})
        for race in event.get("races", []):
            scheduled = race.get("scheduled")
            if scheduled:
                race_date = datetime.fromisoformat(scheduled.replace('Z', '+00:00'))
            else:
                race_date = None
            
            races.append({
                "race_id": race.get("id"),
                "race_name": race.get("name"),
                "race_number": race.get("number"),
                "event_name": event.get("name"),
                "scheduled": scheduled,
                "race_date": race_date,
                "status": race.get("status"),
                "distance_miles": race.get("distance"),
                "laps": race.get("laps"),
                "track_id": track.get("id"),
                "track_name": track.get("name"),
                "track_type": track.get("track_type"),
                "city": track.get("city"),
                "state": track.get("state"),
            })
    
    print(f"  ✓ Found {len(races)} races in {SEASON} schedule")
    return races

def find_upcoming_race(races):
    """Find the next upcoming race"""
    print("\n" + "="*80)
    print("FINDING UPCOMING RACE")
    print("="*80)
    
    now = datetime.now().astimezone()
    
    upcoming = []
    for race in races:
        if race["race_date"] and race["race_date"] > now:
            upcoming.append(race)
    
    # Sort by date
    upcoming = sorted(upcoming, key=lambda x: x["race_date"])
    
    if not upcoming:
        print("  ✗ No upcoming races found")
        return None
    
    next_race = upcoming[0]
    print(f"  ✓ Next race: {next_race['race_name']}")
    print(f"    Date: {next_race['scheduled']}")
    print(f"    Track: {next_race['track_name']} ({next_race['track_type']})")
    print(f"    Race ID: {next_race['race_id']}")
    
    return next_race

# =============================================================================
# FETCH RACE-SPECIFIC DATA FROM SPORTSRADAR
# =============================================================================

def fetch_entry_list(race_id):
    """Fetch entry list for a race"""
    print("\n  Fetching entry list...")
    url = f"{BASE_URL}/{SERIES}/races/{race_id}/entry_list.json?api_key={API_KEY}"
    data = fetch_with_retry(url)
    
    if not data:
        return {}
    
    entries = {}
    for entry in data.get("entry_list", []):
        driver = entry.get("driver", {})
        car = entry.get("car", {})
        manufacturer = car.get("manufacturer", {})
        owner = car.get("owner", {})
        team = car.get("team", {})
        
        driver_id = driver.get("id")
        entries[driver_id] = {
            "driver_id": driver_id,
            "driver_full_name": driver.get("full_name"),
            "driver_first_name": driver.get("first_name"),
            "driver_last_name": driver.get("last_name"),
            "points_eligible": driver.get("points_eligible"),
            "in_chase": driver.get("in_chase"),
            "car_number": car.get("number"),
            "car_id": car.get("id"),
            "manufacturer": manufacturer.get("name"),
            "manufacturer_id": manufacturer.get("id"),
            "team": team.get("name"),
            "team_id": team.get("id"),
            "owner": owner.get("name"),
            "owner_id": owner.get("id"),
            "sponsor": car.get("sponsors"),
            "crew_chief": car.get("crew_chief"),
        }
    
    print(f"    ✓ Found {len(entries)} entries")
    return entries

def fetch_qualifying(race_id):
    """Fetch qualifying results"""
    print("  Fetching qualifying data...")
    url = f"{BASE_URL}/{SERIES}/races/{race_id}/qualifying.json?api_key={API_KEY}"
    data = fetch_with_retry(url)
    
    if not data:
        return {}
    
    qualifying = {}
    for result in data.get("results", []):
        driver = result.get("driver", {})
        driver_id = driver.get("id")
        qualifying[driver_id] = {
            "qualifying_position": result.get("position"),
            "qualifying_speed": result.get("speed"),
            "qualifying_time": result.get("time"),
        }
    
    print(f"    ✓ Found qualifying data for {len(qualifying)} drivers")
    return qualifying

def fetch_starting_grid(race_id):
    """Fetch starting grid"""
    print("  Fetching starting grid...")
    url = f"{BASE_URL}/{SERIES}/races/{race_id}/starting_grid.json?api_key={API_KEY}"
    data = fetch_with_retry(url)
    
    if not data:
        return {}
    
    grid = {}
    for entry in data.get("starting_grid", []):
        driver = entry.get("driver", {})
        driver_id = driver.get("id")
        grid[driver_id] = {
            "start_position": entry.get("position"),  # This is the key feature!
            "grid_position": entry.get("position"),
            "qualifying_lap_time": entry.get("lap_time"),
            "qualifying_lap_speed": entry.get("speed"),
            "grid_status": entry.get("status"),
        }
    
    print(f"    ✓ Found grid data for {len(grid)} drivers")
    return grid

# =============================================================================
# FETCH HISTORICAL DATA FROM DATABASE
# =============================================================================

def fetch_historical_data_from_db(driver_ids, track_name, track_type):
    """Fetch historical data from database for feature calculation"""
    print("\n" + "="*80)
    print("FETCHING HISTORICAL DATA FROM DATABASE")
    print("="*80)
    
    conn = get_db_connection()
    if not conn:
        print("  ✗ Could not connect to database")
        return None
    
    try:
        # Fetch all historical races for these drivers
        driver_ids_str = "', '".join(driver_ids)
        
        query = f"""
        SELECT 
            driver_id,
            driver_full_name,
            race_date,
            track_name,
            track_type,
            team,
            manufacturer,
            finishing_position,
            finish_status,
            start_position,
            avg_speed,
            prior_avg_speed,
            prior_avg_finish_position,
            track_avg_finish_position,
            track_avg_speed,
            track_races_history,
            laps_completed,
            laps_led,
            times_led,
            driver_rating,
            fastest_laps,
            pit_stop_count,
            pit_stop_efficiency,
            points,
            bonus_points,
            best_lap_speed,
            avg_restart_speed
        FROM {DB_TABLE}
        WHERE driver_id IN ('{driver_ids_str}')
        ORDER BY race_date DESC
        """
        
        df = pd.read_sql(query, conn)
        print(f"  ✓ Fetched {len(df)} historical records")
        
        # Also get team and manufacturer stats
        team_query = f"""
        SELECT 
            team,
            AVG(finishing_position) as team_avg_finish_position,
            COUNT(*) as team_races_count
        FROM {DB_TABLE}
        WHERE team IS NOT NULL
        GROUP BY team
        """
        team_df = pd.read_sql(team_query, conn)
        
        manu_query = f"""
        SELECT 
            manufacturer,
            track_type,
            AVG(finishing_position) as manu_track_type_avg
        FROM {DB_TABLE}
        WHERE manufacturer IS NOT NULL AND track_type IS NOT NULL
        GROUP BY manufacturer, track_type
        """
        manu_df = pd.read_sql(manu_query, conn)
        
        conn.close()
        return df, team_df, manu_df
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        conn.close()
        return None, None, None

# =============================================================================
# CALCULATE PREDICTION FEATURES
# =============================================================================

def calculate_form_features(driver_history):
    """Calculate form_last_3, form_last_5, form_last_10 (weighted)"""
    if driver_history is None or len(driver_history) == 0:
        return 20.0, 20.0, 20.0
    
    positions = driver_history['finishing_position'].dropna().values
    
    # form_last_3
    if len(positions) >= 3:
        weights = np.exp(np.linspace(-1, 0, 3))
        form_3 = (positions[:3] * weights / weights.sum()).sum()
    else:
        form_3 = 20.0
    
    # form_last_5
    if len(positions) >= 5:
        weights = np.exp(np.linspace(-1, 0, 5))
        form_5 = (positions[:5] * weights / weights.sum()).sum()
    else:
        form_5 = 20.0
    
    # form_last_10
    if len(positions) >= 10:
        weights = np.exp(np.linspace(-1, 0, 10))
        form_10 = (positions[:10] * weights / weights.sum()).sum()
    else:
        form_10 = 20.0
    
    return form_3, form_5, form_10

def calculate_track_features(driver_history, track_name):
    """Calculate track-specific features"""
    if driver_history is None or len(driver_history) == 0:
        return 20.0, 150.0, 0
    
    track_races = driver_history[driver_history['track_name'] == track_name]
    
    if len(track_races) == 0:
        return 20.0, 150.0, 0
    
    track_avg_finish = track_races['finishing_position'].tail(10).mean()
    track_avg_speed = track_races['avg_speed'].tail(10).mean() if 'avg_speed' in track_races else 150.0
    track_count = len(track_races)
    
    return track_avg_finish, track_avg_speed, track_count

def calculate_win_rate_track_type(driver_history, track_type):
    """Calculate win rate at this track type"""
    if driver_history is None or len(driver_history) == 0:
        return 0.0
    
    track_type_races = driver_history[driver_history['track_type'] == track_type]
    
    if len(track_type_races) == 0:
        return 0.0
    
    wins = (track_type_races['finishing_position'] == 1).sum()
    return wins / len(track_type_races)

def generate_prediction_features(race_info, entries, qualifying, grid, historical_df, team_df, manu_df):
    """Generate all features needed for prediction"""
    print("\n" + "="*80)
    print("GENERATING PREDICTION FEATURES")
    print("="*80)
    
    features_list = []
    
    track_name = race_info['track_name']
    track_type = race_info['track_type']
    
    for driver_id, entry in entries.items():
        driver_name = entry['driver_full_name']
        print(f"  Processing: {driver_name}...", end=" ")
        
        # Get this driver's history
        if historical_df is not None and len(historical_df) > 0:
            driver_history = historical_df[historical_df['driver_id'] == driver_id].sort_values('race_date', ascending=False)
        else:
            driver_history = pd.DataFrame()
        
        # Get qualifying/grid data
        qual_data = qualifying.get(driver_id, {})
        grid_data = grid.get(driver_id, {})
        
        # START POSITION (most critical feature)
        start_position = grid_data.get('start_position') or qual_data.get('qualifying_position') or 20
        
        # Calculate form features
        form_3, form_5, form_10 = calculate_form_features(driver_history)
        
        # Calculate track features
        track_avg_finish, track_avg_speed, track_count = calculate_track_features(driver_history, track_name)
        
        # Win rate at track type
        win_rate = calculate_win_rate_track_type(driver_history, track_type)
        
        # Team momentum (last 5 team races)
        team_momentum = 20.0
        if team_df is not None and entry['team']:
            team_row = team_df[team_df['team'] == entry['team']]
            if len(team_row) > 0:
                team_momentum = team_row['team_avg_finish_position'].values[0]
        
        # Manufacturer track type average
        manu_track_avg = 20.0
        if manu_df is not None and entry['manufacturer']:
            manu_row = manu_df[(manu_df['manufacturer'] == entry['manufacturer']) & 
                               (manu_df['track_type'] == track_type)]
            if len(manu_row) > 0:
                manu_track_avg = manu_row['manu_track_type_avg'].values[0]
        
        # Prior averages from history
        if len(driver_history) > 0:
            prior_avg_finish = driver_history['finishing_position'].mean()
            prior_avg_speed = driver_history['avg_speed'].mean() if 'avg_speed' in driver_history else 150.0
        else:
            prior_avg_finish = 20.0
            prior_avg_speed = 150.0
        
        # Build feature row
        feature_row = {
            # Identifiers
            'race_id': race_info['race_id'],
            'race_name': race_info['race_name'],
            'race_date': race_info['scheduled'],
            'track_name': track_name,
            'track_type': track_type,
            'driver_id': driver_id,
            'driver_full_name': driver_name,
            'car_number': entry['car_number'],
            'team': entry['team'],
            'manufacturer': entry['manufacturer'],
            
            # ========== MODEL FEATURES ==========
            
            # Critical: Start position
            'start_position': float(start_position),
            
            # Form features (weighted recency)
            'form_last_3': float(form_3),
            'form_last_5': float(form_5),
            'form_last_10': float(form_10),
            
            # Track-specific
            'track_avg_finish_position': float(track_avg_finish) if not np.isnan(track_avg_finish) else 20.0,
            'track_avg_speed': float(track_avg_speed) if not np.isnan(track_avg_speed) else 150.0,
            'track_races_history': int(track_count),
            
            # Team & Manufacturer
            'team_momentum': float(team_momentum) if not np.isnan(team_momentum) else 20.0,
            'manu_track_type_avg': float(manu_track_avg) if not np.isnan(manu_track_avg) else 20.0,
            
            # Win rate
            'win_rate_track_type': float(win_rate),
            
            # Prior averages
            'prior_avg_finish_position': float(prior_avg_finish) if not np.isnan(prior_avg_finish) else 20.0,
            'prior_avg_speed': float(prior_avg_speed) if not np.isnan(prior_avg_speed) else 150.0,
            
            # Additional data for reference
            'qualifying_position': qual_data.get('qualifying_position'),
            'qualifying_speed': qual_data.get('qualifying_speed'),
            'qualifying_lap_time': grid_data.get('qualifying_lap_time'),
            'points_eligible': entry['points_eligible'],
            'in_chase': entry['in_chase'],
        }
        
        features_list.append(feature_row)
        print("✓")
    
    return pd.DataFrame(features_list)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*100)
    print("NASCAR UPCOMING RACE PREDICTION DATA FETCHER")
    print(f"Season: {SEASON}")
    print("="*100)
    
    # Step 1: Fetch schedule
    races = fetch_schedule()
    if not races:
        print("\n✗ Failed to fetch schedule. Exiting.")
        return
    
    # Step 2: Find upcoming race
    upcoming_race = find_upcoming_race(races)
    if not upcoming_race:
        print("\n✗ No upcoming race found. Exiting.")
        return
    
    race_id = upcoming_race['race_id']
    
    # Step 3: Fetch race-specific data from SportsRadar
    print("\n" + "="*80)
    print("FETCHING RACE DATA FROM SPORTSRADAR")
    print("="*80)
    
    entries = fetch_entry_list(race_id)
    if not entries:
        print("  ✗ No entry list found. Cannot proceed.")
        return
    
    qualifying = fetch_qualifying(race_id)
    grid = fetch_starting_grid(race_id)
    
    # If no grid data, use qualifying positions as start positions
    if not grid and qualifying:
        print("  ⚠ No starting grid, using qualifying positions")
        grid = {did: {'start_position': q['qualifying_position']} for did, q in qualifying.items()}
    
    # Step 4: Fetch historical data from database
    driver_ids = list(entries.keys())
    historical_df, team_df, manu_df = fetch_historical_data_from_db(
        driver_ids, 
        upcoming_race['track_name'],
        upcoming_race['track_type']
    )
    
    # Step 5: Generate features
    features_df = generate_prediction_features(
        upcoming_race, entries, qualifying, grid,
        historical_df, team_df, manu_df
    )
    
    # Step 6: Save to CSV
    output_file = 'upcoming_race_features.csv'
    features_df.to_csv(output_file, index=False)
    
    print("\n" + "="*100)
    print("OUTPUT SAVED")
    print("="*100)
    print(f"  File: {output_file}")
    print(f"  Drivers: {len(features_df)}")
    print(f"  Features: {len(features_df.columns)}")
    
    # Print feature summary
    print("\n" + "="*100)
    print("FEATURE SUMMARY (for model input)")
    print("="*100)
    
    model_features = [
        'start_position', 'form_last_3', 'form_last_5', 'form_last_10',
        'track_avg_finish_position', 'track_avg_speed', 'track_races_history',
        'team_momentum', 'manu_track_type_avg', 'win_rate_track_type',
        'prior_avg_finish_position', 'prior_avg_speed'
    ]
    
    print(f"\n{'Feature':<30} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("="*70)
    for feat in model_features:
        if feat in features_df.columns:
            print(f"{feat:<30} {features_df[feat].min():>10.2f} {features_df[feat].max():>10.2f} {features_df[feat].mean():>10.2f}")
    
    # Sample output
    print("\n" + "="*100)
    print("SAMPLE DATA (Top 10 by start position)")
    print("="*100)
    sample = features_df.sort_values('start_position').head(10)[
        ['driver_full_name', 'start_position', 'form_last_5', 'track_avg_finish_position', 'team']
    ]
    print(sample.to_string(index=False))
    
    print("\n" + "="*100)
    print("✓ COMPLETE - Ready for prediction!")
    print("="*100)
    
    return features_df

if __name__ == "__main__":
    df = main()