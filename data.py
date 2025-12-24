#!/usr/bin/env python3
"""
NASCAR Multi-Season ML-Ready Dataset Generator
Fetches complete Cup Series data for multiple seasons with pre-match features for elapsed_time prediction
Uses SportsRadar v3 API (nascar-ot3)

ENHANCEMENTS:
- Added entry_list API integration for points_eligible and in_chase status
- Added starting_grid API integration for qualifying lap times and starting positions
- Qualifying lap time is a more precise metric than speed for pre-race performance
- UPDATED: Extracts ALL fields from results.json endpoint (including ALL points fields)
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import datetime
import json

# NOTE: If you get 403 errors, your API key needs NASCAR subscription activated
# Contact SportsRadar support: (610) 233-1333 or support@sportradar.com
API_KEY = "f3iZXDBuhDIoXhYmNtcucblW7xa32KGTtn9aGXZx"
SEASONS = [2024]  # Change this to any years you want
BASE_URL = "https://api.sportradar.us/nascar-ot3"
SERIES = "mc"

# Retry configuration for flaky connections
MAX_RETRIES = 3
RETRY_DELAY = 2

def fetch_with_retry(url, max_retries=MAX_RETRIES):
    """Helper to retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"accept": "application/json"}, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print(f"  ERROR 403: API key may not have NASCAR subscription enabled")
                return None
            elif response.status_code == 404:
                return None
            else:
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return None
    return None

print("="*80)
print(f"NASCAR MULTI-SEASON ML DATASET GENERATOR (UPDATED - ALL RESULTS.JSON FIELDS)")
print(f"Seasons: {', '.join(map(str, SEASONS))}")
print("="*80)

all_race_results = []
season_stats = {}

# ============================================================================
# MAIN LOOP: PROCESS EACH SEASON
# ============================================================================

for season_idx, SEASON in enumerate(SEASONS, 1):
    print(f"\n{'='*80}")
    print(f"[SEASON {season_idx}/{len(SEASONS)}] Processing {SEASON}")
    print(f"{'='*80}")
    
    season_results_count = 0
    
    # ========================================================================
    # STEP 1: FETCH SEASON SCHEDULE
    # ========================================================================
    print(f"\n[STEP 1] Fetching {SEASON} NASCAR schedule...")
    schedule_url = f"{BASE_URL}/{SERIES}/{SEASON}/races/schedule.json?api_key={API_KEY}"

    print(f"  URL: {schedule_url[:80]}...")
    schedule_data = fetch_with_retry(schedule_url)

    if not schedule_data:
        print(f"  ✗ ERROR: Could not fetch schedule for {SEASON}")
        print(f"  Skipping this season...")
        continue

    # Extract race IDs in order
    races_info = []
    race_ids_ordered = []

    for event in schedule_data.get("events", []):
        for race in event.get("races", []):
            race_id = race.get("id")
            races_info.append({
                "race_id": race_id,
                "season": SEASON,
                "event_name": event.get("name"),
                "track_name": event.get("track", {}).get("name"),
                "track_type": event.get("track", {}).get("track_type"),
                "city": event.get("track", {}).get("city"),
                "state": event.get("track", {}).get("state"),
                "race_name": race.get("name"),
                "race_number": race.get("number"),
                "distance_miles": race.get("distance"),
                "laps": race.get("laps"),
                "scheduled_date": race.get("scheduled"),
                "track_id": event.get("track", {}).get("id")
            })
            race_ids_ordered.append(race_id)

    print(f"  ✓ Found {len(races_info)} races for {SEASON}")

    # ========================================================================
    # STEP 2: FETCH QUALIFYING, ENTRY LIST, AND STARTING GRID DATA
    # ========================================================================
    print(f"\n[STEP 2] Fetching qualifying, entry list, and starting grid data for {len(race_ids_ordered)} races...")
    qualifying_data = {}
    entry_list_data = {}
    starting_grid_data = {}  # NEW: Dictionary for starting grid data

    for idx, race_id in enumerate(race_ids_ordered, 1):
        # Fetch qualifying data
        qual_url = f"{BASE_URL}/{SERIES}/races/{race_id}/qualifying.json?api_key={API_KEY}"
        qual_json = fetch_with_retry(qual_url)
        if qual_json:
            for driver in qual_json.get("results", []):
                driver_id = driver.get("driver", {}).get("id")
                qualifying_data[f"{race_id}_{driver_id}"] = {
                    "qual_position": driver.get("position"),
                    "qual_speed": driver.get("speed"),
                    "qual_time": driver.get("time")
                }
        
        # Fetch entry list data for points_eligible status
        entry_url = f"{BASE_URL}/{SERIES}/races/{race_id}/entry_list.json?api_key={API_KEY}"
        entry_json = fetch_with_retry(entry_url)
        if entry_json:
            for entry in entry_json.get("entry_list", []):
                driver = entry.get("driver", {})
                driver_id = driver.get("id")
                entry_list_data[f"{race_id}_{driver_id}"] = {
                    "points_eligible": driver.get("points_eligible"),
                    "in_chase": driver.get("in_chase")
                }
        
        # NEW: Fetch starting grid data
        grid_url = f"{BASE_URL}/{SERIES}/races/{race_id}/starting_grid.json?api_key={API_KEY}"
        grid_json = fetch_with_retry(grid_url)
        if grid_json:
            for entry in grid_json.get("starting_grid", []):
                driver = entry.get("driver", {})
                driver_id = driver.get("id")
                car = entry.get("car", {})
                # Extract starting position and qualifying lap time from grid
                starting_grid_data[f"{race_id}_{driver_id}"] = {
                    "grid_position": entry.get("position"),
                    "qual_lap_time": entry.get("lap_time"),  # Precise lap time in seconds
                    "qual_lap_speed": entry.get("speed"),    # Alternative speed metric
                    "grid_status": entry.get("status"),
                    "car_number": car.get("number")
                }
        
        time.sleep(0.3)

    print(f"  ✓ Fetched qualifying data for {len(qualifying_data)} driver-race combinations")
    print(f"  ✓ Fetched entry list data for {len(entry_list_data)} driver-race combinations")
    print(f"  ✓ Fetched starting grid data for {len(starting_grid_data)} driver-race combinations")  # NEW

    # ========================================================================
    # STEP 3: INITIALIZE ROLLING STATS TRACKERS (PER SEASON)
    # ========================================================================
    print(f"\n[STEP 3] Initializing rolling statistics trackers...")

    driver_race_history = defaultdict(list)  # driver_id -> list of race results
    driver_track_history = defaultdict(lambda: defaultdict(list))  # driver_id -> track_id -> results
    team_race_stats = defaultdict(list)  # team_id -> list of race stats
    manufacturer_race_stats = defaultdict(list)  # manufacturer_id -> list of race stats

    season_race_results = []

    # ========================================================================
    # STEP 4: PROCESS RACES IN CHRONOLOGICAL ORDER
    # ========================================================================
    print(f"\n[STEP 4] Processing {len(race_ids_ordered)} races with rolling features...")

    for race_num, race_id in enumerate(race_ids_ordered, 1):
        race_info = races_info[race_num - 1]
        track_id = race_info["track_id"]
        track_name = race_info["track_name"]
        team_id = None
        
        # Fetch race results
        results_url = f"{BASE_URL}/{SERIES}/races/{race_id}/results.json?api_key={API_KEY}"
        
        results_data = fetch_with_retry(results_url)
        
        if not results_data:
            print(f"  [{race_num:2d}/{len(race_ids_ordered)}] (skipped - no data)")
            time.sleep(0.5)
            continue
        
        try:
            race_name = results_data.get("name", "Unknown")
            results_list = results_data.get("results", [])
            
            print(f"  [{race_num:2d}/{len(race_ids_ordered)}] {race_name:50s} ... ", end="", flush=True)
            
            for result in results_list:
                driver = result.get("driver", {})
                car = result.get("car", {})
                manufacturer = car.get("manufacturer", {})
                owner = car.get("owner", {})
                team = car.get("team", {})
                
                driver_id = driver.get("id")
                car_number = car.get("number")
                manufacturer_id = manufacturer.get("id")
                team_id = team.get("id")
                
                # ================================================================
                # CALCULATE ROLLING FEATURES FROM PRIOR RACES
                # ================================================================
                
                # Prior races for this driver (all races before current)
                prior_races = driver_race_history.get(driver_id, [])
                
                # Rolling averages (from prior races only)
                prior_avg_speed = None
                prior_avg_position = None
                prior_dnf_rate = None
                prior_avg_laps_completed = None
                prior_avg_pit_stops = None
                
                if len(prior_races) > 0:
                    speeds = [r["avg_speed"] for r in prior_races if r.get("avg_speed")]
                    positions = [r["finishing_position"] for r in prior_races if r.get("finishing_position")]
                    dnf_count = sum(1 for r in prior_races if r.get("status") not in ["running", "completed"])
                    laps = [r["laps_completed"] for r in prior_races if r.get("laps_completed")]
                    pit_counts = [r["pit_stop_count"] for r in prior_races if r.get("pit_stop_count")]
                    
                    prior_avg_speed = np.mean(speeds) if speeds else None
                    prior_avg_position = np.mean(positions) if positions else None
                    prior_dnf_rate = dnf_count / len(prior_races)
                    prior_avg_laps_completed = np.mean(laps) if laps else None
                    prior_avg_pit_stops = np.mean(pit_counts) if pit_counts else None
                
                # Track-specific performance (how driver performed at this track historically)
                track_races = driver_track_history[driver_id].get(track_id, [])
                track_avg_speed = None
                track_avg_position = None
                track_races_count = len(track_races)
                
                if len(track_races) > 0:
                    track_speeds = [r["avg_speed"] for r in track_races if r.get("avg_speed")]
                    track_positions = [r["finishing_position"] for r in track_races if r.get("finishing_position")]
                    track_avg_speed = np.mean(track_speeds) if track_speeds else None
                    track_avg_position = np.mean(track_positions) if track_positions else None
                
                # Team aggregate performance
                team_races = team_race_stats.get(team_id, []) if team_id else []
                team_avg_speed = None
                team_avg_position = None
                if len(team_races) > 0:
                    team_speeds = [r["avg_speed"] for r in team_races if r.get("avg_speed")]
                    team_positions = [r["finishing_position"] for r in team_races if r.get("finishing_position")]
                    team_avg_speed = np.mean(team_speeds) if team_speeds else None
                    team_avg_position = np.mean(team_positions) if team_positions else None
                
                # Manufacturer aggregate performance
                mfg_races = manufacturer_race_stats.get(manufacturer_id, []) if manufacturer_id else []
                mfg_avg_speed = None
                mfg_avg_position = None
                if len(mfg_races) > 0:
                    mfg_speeds = [r["avg_speed"] for r in mfg_races if r.get("avg_speed")]
                    mfg_positions = [r["finishing_position"] for r in mfg_races if r.get("finishing_position")]
                    mfg_avg_speed = np.mean(mfg_speeds) if mfg_speeds else None
                    mfg_avg_position = np.mean(mfg_positions) if mfg_positions else None
                
                # ================================================================
                # PIT STOP EFFICIENCY METRICS
                # ================================================================
                pit_stops = result.get("pit_stops", [])
                
                pit_stop_efficiency = None
                total_pit_time = 0
                total_positions_gained = 0
                
                if len(pit_stops) > 0:
                    for pit_stop in pit_stops:
                        # Ignore dummy pit stops (sequence 1-2 often have 0 times)
                        if pit_stop.get("sequence", 0) >= 3:
                            in_time = pit_stop.get("in_time", 0)
                            out_time = pit_stop.get("out_time", 0)
                            if out_time > 0 and in_time >= 0:
                                pit_duration = out_time - in_time
                                total_pit_time += pit_duration
                            positions_gained = pit_stop.get("positions_gained", 0)
                            if positions_gained:
                                total_positions_gained += positions_gained
                    
                    # Efficiency: positions gained per second in pit (higher is better)
                    if total_pit_time > 0:
                        pit_stop_efficiency = total_positions_gained / total_pit_time
                
                # ================================================================
                # HANDLE DNF: SET ELAPSED TIME TO 0
                # ================================================================
                status = result.get("status", "")
                elapsed_time = result.get("elapsed_time", 0) if result.get("elapsed_time") else 0
                
                # If DNF or didn't finish running, set elapsed_time to 0
                if status not in ["running", "completed"]:
                    elapsed_time = 0
                
                # ================================================================
                # FETCH POINTS ELIGIBLE AND IN_CHASE DATA
                # ================================================================
                entry_key = f"{race_id}_{driver_id}"
                points_eligible = entry_list_data.get(entry_key, {}).get("points_eligible")
                in_chase = entry_list_data.get(entry_key, {}).get("in_chase")
                
                # ================================================================
                # NEW: FETCH STARTING GRID DATA (QUALIFYING DETAILS)
                # ================================================================
                grid_key = f"{race_id}_{driver_id}"
                grid_position = starting_grid_data.get(grid_key, {}).get("grid_position")
                qual_lap_time = starting_grid_data.get(grid_key, {}).get("qual_lap_time")
                qual_lap_speed = starting_grid_data.get(grid_key, {}).get("qual_lap_speed")
                
                # ================================================================
                # BUILD COMPLETE FEATURE ROW
                # ================================================================
                driver_row = {
                    # Identity
                    "race_id": race_id,
                    "race_number": race_info["race_number"],
                    "race_name": race_name,
                    "season": SEASON,
                    "race_date": race_info["scheduled_date"],
                    
                    # Track info
                    "track_id": track_id,
                    "track_name": track_name,
                    "track_type": race_info["track_type"],
                    "track_distance_miles": race_info["distance_miles"],
                    "track_laps": race_info["laps"],
                    
                    # Driver info
                    "driver_id": driver_id,
                    "driver_first_name": driver.get("first_name"),
                    "driver_last_name": driver.get("last_name"),
                    "driver_full_name": driver.get("full_name"),
                    "car_number": car_number,
                    
                    # Car/Team info
                    "car_id": car.get("id"),
                    "sponsor": car.get("sponsors"),
                    "crew_chief": car.get("crew_chief"),
                    "manufacturer": manufacturer.get("name"),
                    "manufacturer_id": manufacturer_id,
                    "owner": owner.get("name"),
                    "owner_id": owner.get("id"),
                    "team": team.get("name"),
                    "team_id": team_id,
                    
                    # ============ PRE-MATCH DATA ============
                    
                    # Qualifying data
                    "qualifying_position": qualifying_data.get(f"{race_id}_{driver_id}", {}).get("qual_position"),
                    "qualifying_speed": qualifying_data.get(f"{race_id}_{driver_id}", {}).get("qual_speed"),
                    "qualifying_time": qualifying_data.get(f"{race_id}_{driver_id}", {}).get("qual_time"),
                    
                    # NEW: Starting Grid Data (Enhanced Qualifying Metrics)
                    "grid_position": grid_position,
                    "qualifying_lap_time": qual_lap_time,  # Precise time in seconds
                    "qualifying_lap_speed": qual_lap_speed,  # Speed from grid data
                    
                    # Entry List Data (Championship Eligibility)
                    "points_eligible": points_eligible,
                    "in_chase": in_chase,
                    
                    # Prior season rolling stats
                    "prior_races_count": len(prior_races),
                    "prior_avg_speed": prior_avg_speed,
                    "prior_avg_finish_position": prior_avg_position,
                    "prior_dnf_rate": prior_dnf_rate,
                    "prior_avg_laps_completed": prior_avg_laps_completed,
                    "prior_avg_pit_stops": prior_avg_pit_stops,
                    "prior_speed_consistency": np.std([r["avg_speed"] for r in prior_races if r.get("avg_speed")]) if prior_races else None,
                    
                    # Track-specific historical performance
                    "track_races_history": track_races_count,
                    "track_avg_speed": track_avg_speed,
                    "track_avg_finish_position": track_avg_position,
                    
                    # Team aggregate
                    "team_races_count": len(team_races),
                    "team_avg_speed": team_avg_speed,
                    "team_avg_finish_position": team_avg_position,
                    
                    # Manufacturer aggregate
                    "manufacturer_races_count": len(mfg_races),
                    "manufacturer_avg_speed": mfg_avg_speed,
                    "manufacturer_avg_finish_position": mfg_avg_position,
                    
                    # ============ RACE RESULTS ============
                    
                    # Starting & Finishing
                    "start_position": result.get("start_position"),
                    "finishing_position": result.get("position"),
                    "finish_status": status,
                    
                    # Laps & Leads
                    "laps_completed": result.get("laps_completed"),
                    "laps_led": result.get("laps_led"),
                    "times_led": result.get("times_led"),
                    
                    # Performance metrics
                    "avg_speed": result.get("avg_speed"),
                    "avg_position": result.get("avg_position"),
                    "best_lap_number": result.get("best_lap"),
                    "best_lap_speed": result.get("best_lap_speed"),
                    "best_lap_time": result.get("best_lap_time"),
                    "last_lap_speed": result.get("last_lap_speed"),
                    "last_lap_time": result.get("last_lap_time"),
                    "avg_restart_speed": result.get("avg_restart_speed"),
                    "driver_rating": result.get("driver_rating"),
                    "fastest_laps": result.get("fastest_laps"),
                    
                    # Pit stops
                    "pit_stop_count": result.get("pit_stop_count"),
                    "pit_stop_efficiency": pit_stop_efficiency,
                    "pit_total_positions_change": total_positions_gained,
                    
                    # Points & Scoring (ALL FIELDS FROM RESULTS.JSON API)
                    "points": result.get("points"),
                    "bonus_points": result.get("bonus_points"),
                    "penalty_points": result.get("penalty_points"),
                    "fastest_lap_points": result.get("fastest_lap_points"),
                    "stage_1_points": result.get("stage_1_points"),
                    "stage_2_points": result.get("stage_2_points"),
                    "stage_1_win": result.get("stage_1_win"),
                    "stage_2_win": result.get("stage_2_win"),
                    "money": result.get("money"),
                    
                    # ============ TARGET VARIABLE ============
                    "elapsed_time_seconds": elapsed_time,  # 0 if DNF
                }
                
                season_race_results.append(driver_row)
                season_results_count += 1
                
                # ================================================================
                # UPDATE ROLLING STATS FOR FUTURE RACES
                # ================================================================
                
                # Store this race result for future rolling calculations
                race_result_for_history = {
                    "race_id": race_id,
                    "race_number": race_info["race_number"],
                    "track_id": track_id,
                    "finishing_position": result.get("position"),
                    "status": status,
                    "avg_speed": result.get("avg_speed"),
                    "laps_completed": result.get("laps_completed"),
                    "pit_stop_count": result.get("pit_stop_count"),
                    "laps_led": result.get("laps_led"),
                    "points": result.get("points"),
                    "elapsed_time": elapsed_time,
                }
                
                driver_race_history[driver_id].append(race_result_for_history)
                driver_track_history[driver_id][track_id].append(race_result_for_history)
                
                if team_id:
                    team_race_stats[team_id].append(race_result_for_history)
                if manufacturer_id:
                    manufacturer_race_stats[manufacturer_id].append(race_result_for_history)
            
            print(f"✓ ({len(results_list)} drivers)")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            time.sleep(0.5)
    
    # ========================================================================
    # COMBINE SEASON DATA
    # ========================================================================
    all_race_results.extend(season_race_results)
    season_stats[SEASON] = {
        "total_rows": season_results_count,
        "races": len([r for r in season_race_results if r.get("race_number") == 1]),  # Approximate
    }
    
    print(f"\n  ✓ {SEASON}: {season_results_count} driver-race records added")

# ============================================================================
# STEP 5: BUILD AND SAVE COMBINED DATASET
# ============================================================================
print(f"\n{'='*80}")
print(f"[FINAL STEP] Building combined dataset for {len(SEASONS)} season(s)...")
print(f"{'='*80}")

df_results = pd.DataFrame(all_race_results)

# Reorder columns for readability
priority_cols = [
    "race_id", "race_number", "race_name", "season", "race_date",
    "track_name", "track_type", "track_distance_miles", "track_laps",
    "driver_full_name", "car_number", "team",
    "qualifying_position", "qualifying_speed", "qualifying_time",
    "grid_position", "qualifying_lap_time", "qualifying_lap_speed",  # Grid data
    "points_eligible", "in_chase",  # Championship eligibility
    "start_position", "finishing_position", "finish_status",
    "prior_races_count", "prior_avg_speed", "prior_avg_finish_position", 
    "prior_dnf_rate", "prior_speed_consistency",
    "track_races_history", "track_avg_speed", "track_avg_finish_position",
    "team_avg_speed", "team_avg_finish_position",
    "manufacturer_avg_speed", "manufacturer_avg_finish_position",
    "avg_speed", "avg_position", "laps_completed", "laps_led", "times_led",
    "best_lap_number", "best_lap_speed", "best_lap_time",
    "last_lap_speed", "last_lap_time", "avg_restart_speed",
    "driver_rating", "fastest_laps",
    "pit_stop_count", "pit_stop_efficiency",
    # ALL POINT FIELDS FROM RESULTS.JSON
    "points", "bonus_points", "penalty_points", "fastest_lap_points",
    "stage_1_points", "stage_2_points", "stage_1_win", "stage_2_win",
    "money",
    "elapsed_time_seconds"
]

# Only keep priority cols that exist
priority_cols = [c for c in priority_cols if c in df_results.columns]
other_cols = [c for c in df_results.columns if c not in priority_cols]

df_results = df_results[priority_cols + other_cols]

# Save main dataset
season_str = "_".join(map(str, SEASONS))
output_file = f"NASCAR_{season_str}_ML_Dataset.csv"
df_results.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"✓ DATASET SAVED: {output_file}")
print(f"{'='*80}")
print(f"  Total Rows: {len(df_results):,} (driver-race combinations)")
print(f"  Total Columns: {len(df_results.columns)}")
print(f"  Seasons: {', '.join(map(str, sorted(df_results['season'].unique())))}")
print(f"  Total Races: {df_results['race_number'].nunique() * len(SEASONS)}")
print(f"  Unique Drivers: {df_results['driver_full_name'].nunique()}")
print(f"\n  Breakdown by Season:")
for season in sorted(SEASONS):
    season_data = df_results[df_results['season'] == season]
    dnf_count = (season_data['elapsed_time_seconds'] == 0).sum()
    completed = (season_data['elapsed_time_seconds'] > 0).sum()
    print(f"    {season}: {len(season_data):5d} records ({completed:4d} completed, {dnf_count:3d} DNF)")

print(f"\n  Target Variable: elapsed_time_seconds")
dnf_total = (df_results['elapsed_time_seconds'] == 0).sum()
completed_total = (df_results['elapsed_time_seconds'] > 0).sum()
print(f"    Total DNF (elapsed_time=0): {dnf_total:,}")
print(f"    Total Completed: {completed_total:,}")
print(f"    Mean elapsed time: {df_results[df_results['elapsed_time_seconds'] > 0]['elapsed_time_seconds'].mean():.1f}s")

# Points breakdown
print(f"\n  FEATURE: Points (from results.json)")
print(f"    Mean points: {df_results['points'].mean():.2f}")
print(f"    Min points: {df_results['points'].min()}")
print(f"    Max points: {df_results['points'].max()}")

print(f"\n  FEATURE: Bonus Points (from results.json)")
bonus_valid = df_results['bonus_points'].notna().sum()
print(f"    Valid bonus points records: {bonus_valid:,}")
print(f"    Mean bonus points: {df_results['bonus_points'].mean():.2f}")

print(f"\n  FEATURE: Penalty Points (from results.json)")
penalty_valid = df_results['penalty_points'].notna().sum()
print(f"    Valid penalty points records: {penalty_valid:,}")
print(f"    Mean penalty points: {df_results['penalty_points'].mean():.2f}")

print(f"\n  ALL RESULTS.JSON FIELDS EXTRACTED:")
results_json_fields = ["points", "bonus_points", "penalty_points", "fastest_lap_points",
                       "stage_1_points", "stage_2_points", "stage_1_win", "stage_2_win",
                       "money", "driver_rating", "fastest_laps", "laps_led", "times_led",
                       "best_lap_number", "best_lap_speed", "best_lap_time",
                       "last_lap_speed", "last_lap_time", "avg_restart_speed"]
for field in results_json_fields:
    if field in df_results.columns:
        print(f"    ✓ {field}")

print(f"\n{'='*80}")
print("DATA QUALITY CHECK")
print("="*80)
print(f"\nMissing Values (top 15 columns):")
missing = df_results.isnull().sum().sort_values(ascending=False).head(15)
for col, count in missing.items():
    pct = (count / len(df_results)) * 100
    print(f"  {col:40s}: {count:6d} ({pct:5.1f}%)")

print(f"\nElapsed Time Distribution (All Seasons Combined):")
print(df_results['elapsed_time_seconds'].describe())

print(f"\n{'='*80}")
print("✅ COMPLETE - ALL RESULTS.JSON FIELDS STORED TO CSV")
print("="*80)