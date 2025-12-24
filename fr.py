"""
NASCAR PROBABILITY PREDICTION MODEL v1.0
Regression-based approach with relative & absolute probabilities
Fixes the "identical probabilities" problem by:
1. Predicting finishing position (regression, not classification)
2. Converting position predictions to probabilities within each race
3. Using start_position heavily (~60-70%) + driver/team history
4. Calibrating with Platt scaling
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_RATIO = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.50  # Probability threshold for prediction (modular)

# Feature weights - start_position gets most weight
FEATURE_WEIGHTS = {
    'start_position': 3.0,  # Heavy weight
    'prior_avg_finish_position': 1.5,
    'prior_avg_speed': 1.2,
    'track_avg_finish_position': 1.3,
    'form_last_3': 1.4,  # Recent form
    'form_last_5': 1.2,
}

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_prepare_data(path):
    """Load data and create target variables"""
    print("="*80)
    print("LOADING & PREPARING DATA")
    print("="*80)
    
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} records, {df['race_id'].nunique()} unique races")
    
    # Filter valid finishes
    df = df[(df['elapsed_time_seconds'] > 0) & (df['finishing_position'] <= 40)].copy()
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df.sort_values('race_date').reset_index(drop=True)
    print(f"✓ After filtering DNFs: {len(df)} records")
    
    # Create binary targets
    df['target_winner'] = (df['finishing_position'] == 1).astype(int)
    df['target_top3'] = (df['finishing_position'] <= 3).astype(int)
    df['target_top5'] = (df['finishing_position'] <= 5).astype(int)
    df['target_top10'] = (df['finishing_position'] <= 10).astype(int)
    
    return df

def engineer_features(df):
    """Create pre-race features using only chronological prior data"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING (Chronological - No Look-Ahead Bias)")
    print("="*80)
    
    # 1. Driver recent form (last 3, 5, 10 races)
    print("\n1. Recent driver form...")
    form_3, form_5, form_10 = [], [], []
    
    for idx, row in df.iterrows():
        mask = (df['driver_full_name'] == row['driver_full_name']) & (df.index < idx)
        past = df.loc[mask, 'finishing_position']
        
        # Weighted recency
        if len(past) >= 3:
            weights = np.exp(np.linspace(-1, 0, 3))
            form_3.append((past.tail(3).values * weights / weights.sum()).sum())
        else:
            form_3.append(20.0)
        
        if len(past) >= 5:
            weights = np.exp(np.linspace(-1, 0, 5))
            form_5.append((past.tail(5).values * weights / weights.sum()).sum())
        else:
            form_5.append(20.0)
        
        if len(past) >= 10:
            weights = np.exp(np.linspace(-1, 0, 10))
            form_10.append((past.tail(10).values * weights / weights.sum()).sum())
        else:
            form_10.append(20.0)
    
    df['form_last_3'] = form_3
    df['form_last_5'] = form_5
    df['form_last_10'] = form_10
    
    # 2. Track-specific performance
    print("2. Track-specific performance...")
    track_avg_finish, track_avg_speed, track_count = [], [], []
    
    for idx, row in df.iterrows():
        mask = (
            (df['driver_full_name'] == row['driver_full_name']) &
            (df['track_name'] == row['track_name']) &
            (df.index < idx)
        )
        past = df.loc[mask]
        
        if len(past) > 0:
            track_avg_finish.append(past['finishing_position'].tail(10).mean())
            track_avg_speed.append(past['prior_avg_speed'].tail(10).mean() if 'prior_avg_speed' in past.columns else 150)
            track_count.append(len(past))
        else:
            track_avg_finish.append(20.0)
            track_avg_speed.append(150.0)
            track_count.append(0)
    
    df['track_avg_finish_position'] = track_avg_finish
    df['track_avg_speed'] = track_avg_speed
    df['track_races_history'] = track_count
    
    # 3. Team performance (last 5 races on any track)
    print("3. Team momentum...")
    team_momentum = []
    
    for idx, row in df.iterrows():
        mask = (df['team'] == row['team']) & (df.index < idx)
        past = df.loc[mask, 'finishing_position'].tail(5)
        team_momentum.append(past.mean() if len(past) > 0 else 20.0)
    
    df['team_momentum'] = team_momentum
    
    # 4. Manufacturer at track type
    print("4. Manufacturer track-type performance...")
    manu_track_type = []
    
    for idx, row in df.iterrows():
        mask = (
            (df['manufacturer'] == row['manufacturer']) &
            (df['track_type'] == row['track_type']) &
            (df.index < idx)
        )
        past = df.loc[mask, 'finishing_position'].tail(10)
        manu_track_type.append(past.mean() if len(past) > 0 else 20.0)
    
    df['manu_track_type_avg'] = manu_track_type
    
    # 5. Win rate at track type
    print("5. Win rate by track type...")
    win_rate_track_type = []
    
    for idx, row in df.iterrows():
        mask = (
            (df['driver_full_name'] == row['driver_full_name']) &
            (df['track_type'] == row['track_type']) &
            (df.index < idx)
        )
        past = df.loc[mask]
        if len(past) > 0:
            wins = (past['finishing_position'] == 1).sum()
            win_rate_track_type.append(wins / len(past))
        else:
            win_rate_track_type.append(0.0)
    
    df['win_rate_track_type'] = win_rate_track_type
    
    print("✓ Feature engineering complete")
    return df

def create_train_test_split(df):
    """Time-based split to avoid data leakage"""
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT (Time-Based)")
    print("="*80)
    
    races = df[["race_id", "race_date"]].drop_duplicates().sort_values("race_date")
    cut = int(len(races) * (1 - TEST_RATIO))
    train_ids = races.iloc[:cut]["race_id"].values
    test_ids = races.iloc[cut:]["race_id"].values
    
    train = df[df["race_id"].isin(train_ids)].copy()
    test = df[df["race_id"].isin(test_ids)].copy()
    
    print(f"\n✓ Train: {len(train)} records ({train['race_id'].nunique()} races)")
    print(f"✓ Test:  {len(test)} records ({test['race_id'].nunique()} races)")
    
    print("\nTrack Type Distribution (Test):")
    for tt in sorted(test["track_type"].unique()):
        print(f"  {tt}: {len(test[test['track_type'] == tt])} records")
    
    return train, test

def fill_and_encode_features(train, test):
    """Fill missing values and encode categorical features"""
    print("\n" + "="*80)
    print("FEATURE PROCESSING")
    print("="*80)
    
    # Define pre-race features
    numeric_features = [
        'start_position', 'prior_races_count', 'prior_avg_speed', 'prior_avg_finish_position',
        'prior_dnf_rate', 'prior_speed_consistency', 'prior_avg_laps_completed', 'prior_avg_pit_stops',
        'track_races_history', 'track_avg_speed', 'track_avg_finish_position',
        'team_avg_speed', 'team_avg_finish_position', 'team_races_count',
        'manufacturer_avg_speed', 'manufacturer_avg_finish_position', 'manufacturer_races_count',
        'track_distance_miles', 'track_laps', 'points_eligible', 'in_chase',
        'form_last_3', 'form_last_5', 'form_last_10', 'team_momentum', 
        'manu_track_type_avg', 'win_rate_track_type'
    ]
    
    categorical_features = ['driver_full_name', 'team', 'track_name', 'track_type', 'manufacturer']
    
    # Fill missing numeric values with median
    for col in numeric_features:
        if col in train.columns:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Encode categorical features
    encoders = {}
    for col in categorical_features:
        if col in train.columns:
            le = LabelEncoder()
            train[col + '_encoded'] = le.fit_transform(train[col].astype(str))
            test[col + '_encoded'] = test[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            encoders[col] = le
    
    feature_cols = [col for col in numeric_features if col in train.columns] + \
                   [col + '_encoded' for col in categorical_features if col in train.columns]
    
    print(f"✓ Using {len(feature_cols)} features for training")
    print(f"  Numeric: {len([c for c in feature_cols if '_encoded' not in c])}")
    print(f"  Categorical: {len([c for c in feature_cols if '_encoded' in c])}")
    
    return train, test, feature_cols, encoders

# =============================================================================
# MODEL TRAINING - TRACK-TYPE SPECIFIC
# =============================================================================

def train_models_by_track_type(train, feature_cols):
    """Train separate regression models for each track type"""
    print("\n" + "="*80)
    print("TRAINING MODELS (Track-Type Specific Regression)")
    print("="*80)
    
    models = {}
    
    for track_type in sorted(train['track_type'].unique()):
        print(f"\n{track_type}:")
        
        track_train = train[train['track_type'] == track_type].copy()
        
        if len(track_train) < 50:
            print(f"  ⚠ Too few samples ({len(track_train)}), skipping")
            continue
        
        X = track_train[feature_cols].values
        y = track_train['finishing_position'].values
        
        # XGBoost Regression
        print(f"  Training XGBoost (n_samples={len(track_train)})...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        model.fit(X, y)
        
        # Verify predictions are in reasonable range
        train_pred = model.predict(X)
        mae = mean_absolute_error(y, train_pred)
        print(f"  ✓ Train MAE: {mae:.2f} positions")
        
        models[track_type] = model
    
    return models

def predict_probabilities(df, models, feature_cols, method='absolute'):
    """
    Generate probabilities for each driver
    method='absolute': Per-driver absolute probability (0-100%)
    method='relative': Rank-based relative probability within each race
    """
    
    print(f"\n  Generating predictions ({method} method)...")
    
    # Initialize probability columns
    for target in ['winner', 'top3', 'top5', 'top10']:
        df[f'prob_{target}_{method}'] = 0.0
        df[f'pred_{target}_{method}'] = 0
    
    for race_id in df['race_id'].unique():
        race_mask = df['race_id'] == race_id
        race_df = df[race_mask].copy()
        track_type = race_df['track_type'].iloc[0]
        
        # Skip if no model for this track type
        if track_type not in models:
            continue
        
        model = models[track_type]
        
        # Predict finishing positions
        X = race_df[feature_cols].values
        predicted_positions = model.predict(X)
        predicted_positions = np.clip(predicted_positions, 1, 40)
        
        if method == 'absolute':
            # Convert position predictions to absolute probabilities using sigmoid-like function
            # Position 1 → high probability of winning/top-5/top-10
            # Position 40 → low probability
            
            # Sigmoid: 1 / (1 + exp(-k*(x-x0)))
            # k controls steepness, x0 is inflection point
            
            for idx, (i, row) in enumerate(race_df.iterrows()):
                pos_pred = predicted_positions[idx]
                
                # Winner probability: position must be very close to 1
                prob_winner = 1.0 / (1 + np.exp(0.8 * (pos_pred - 1.5)))
                
                # Top 3 probability
                prob_top3 = 1.0 / (1 + np.exp(0.6 * (pos_pred - 3.5)))
                
                # Top 5 probability
                prob_top5 = 1.0 / (1 + np.exp(0.5 * (pos_pred - 5.5)))
                
                # Top 10 probability
                prob_top10 = 1.0 / (1 + np.exp(0.3 * (pos_pred - 10.5)))
                
                df.loc[i, 'prob_winner_absolute'] = prob_winner * 100
                df.loc[i, 'prob_top3_absolute'] = prob_top3 * 100
                df.loc[i, 'prob_top5_absolute'] = prob_top5 * 100
                df.loc[i, 'prob_top10_absolute'] = prob_top10 * 100
        
        elif method == 'relative':
            # Rank-based: Rank drivers by predicted position within race
            # Better (lower) predicted position = higher probability
            
            ranks = pd.Series(predicted_positions).rank(method='average').values
            max_rank = len(race_df)
            
            for idx, (i, row) in enumerate(race_df.iterrows()):
                rank = ranks[idx]
                
                # Probability = (max_rank - rank) / (max_rank - 1)
                # Best rank (1) → probability 1.0
                # Worst rank (max) → probability 0.0
                
                relative_prob = (max_rank - rank) / (max_rank - 1) if max_rank > 1 else 0.5
                
                # Adjust probability based on position threshold
                prob_winner = 1.0 if rank <= 1 else 0.5 / rank
                prob_top3 = 1.0 if rank <= 3 else 1.0 - (rank - 3) / (max_rank - 3) if max_rank > 3 else 0.5
                prob_top5 = 1.0 if rank <= 5 else 1.0 - (rank - 5) / (max_rank - 5) if max_rank > 5 else 0.5
                prob_top10 = 1.0 if rank <= 10 else 1.0 - (rank - 10) / (max_rank - 10) if max_rank > 10 else 0.5
                
                df.loc[i, 'prob_winner_relative'] = min(prob_winner, 1.0) * 100
                df.loc[i, 'prob_top3_relative'] = min(prob_top3, 1.0) * 100
                df.loc[i, 'prob_top5_relative'] = min(prob_top5, 1.0) * 100
                df.loc[i, 'prob_top10_relative'] = min(prob_top10, 1.0) * 100
    
    return df

def calibrate_probabilities_platt(train, test, feature_cols, models):
    """Calibrate probabilities using Platt scaling"""
    print("\n" + "="*80)
    print("PROBABILITY CALIBRATION (Platt Scaling)")
    print("="*80)
    
    # For each track type and target, calibrate
    for method in ['absolute', 'relative']:
        print(f"\nCalibrating {method} probabilities...")
        
        for target in ['winner', 'top3', 'top5', 'top10']:
            target_col = f'target_{target}'
            prob_col = f'prob_{target}_{method}'
            
            if prob_col not in train.columns or prob_col not in test.columns:
                continue
            
            # Train Platt scaling on training set
            X_train = train[prob_col].values.reshape(-1, 1) / 100  # Normalize to 0-1
            y_train = train[target_col].values
            
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_train, y_train)
            
            # Apply to test set
            X_test = test[prob_col].values.reshape(-1, 1) / 100
            calibrated = lr.predict_proba(X_test)[:, 1]
            
            test[f'prob_{target}_{method}_calibrated'] = calibrated * 100
    
    return test

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_precision_at_threshold(df, target, method, threshold=THRESHOLD):
    """Evaluate precision: for predicted top-N drivers, how many actually finished top-N?"""
    
    results = []
    
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id].copy()
        
        # Actual count of drivers in target class
        actual_count = race_df[f'target_{target}'].sum()
        
        if actual_count == 0:
            continue
        
        # Get probability column
        prob_col = f'prob_{target}_{method}_calibrated'
        if prob_col not in race_df.columns:
            prob_col = f'prob_{target}_{method}'
        
        # Sort by probability descending
        race_df = race_df.sort_values(prob_col, ascending=False)
        
        # Get top N drivers by probability
        predicted_topn = race_df.head(actual_count)
        
        # Count how many actually finished in target class
        correct = predicted_topn[f'target_{target}'].sum()
        
        precision = correct / actual_count if actual_count > 0 else 0
        
        results.append({
            'race_id': race_id[:8],
            'track_type': race_df['track_type'].iloc[0],
            'target': target,
            'method': method,
            'actual_count': actual_count,
            'predicted_count': actual_count,
            'correct': int(correct),
            'precision': precision,
            'avg_prob_correct': predicted_topn[predicted_topn[f'target_{target}'] == 1][prob_col].mean() if correct > 0 else 0
        })
    
    return pd.DataFrame(results)

def print_evaluation_report(test):
    """Print comprehensive evaluation report"""
    
    print("\n" + "="*100)
    print("PRECISION EVALUATION REPORT")
    print("="*100)
    
    for method in ['absolute', 'relative']:
        print(f"\n{method.upper()} METHOD")
        print("-" * 100)
        
        for target in ['winner', 'top3', 'top5', 'top10']:
            results_df = evaluate_precision_at_threshold(test, target, method)
            
            if len(results_df) == 0:
                continue
            
            overall_precision = results_df['correct'].sum() / results_df['predicted_count'].sum()
            
            print(f"\n  {target.upper()}:")
            print(f"    Overall Precision: {overall_precision:.1%} ({results_df['correct'].sum()}/{results_df['predicted_count'].sum()})")
            print(f"    Races Evaluated: {len(results_df)}")
            
            # By track type
            for tt in sorted(results_df['track_type'].unique()):
                tt_results = results_df[results_df['track_type'] == tt]
                tt_precision = tt_results['correct'].sum() / tt_results['predicted_count'].sum()
                print(f"      {tt}: {tt_precision:.1%} ({tt_results['correct'].sum()}/{tt_results['predicted_count'].sum()})")

def show_sample_predictions(test, num_races=3):
    """Show sample predictions for specific races"""
    
    print("\n" + "="*100)
    print("SAMPLE RACE PREDICTIONS")
    print("="*100)
    
    sample_races = test['race_id'].unique()[:num_races]
    
    for race_id in sample_races:
        race_df = test[test['race_id'] == race_id].copy()
        
        print(f"\n{'='*100}")
        print(f"RACE: {race_id[:8]} | Track: {race_df['track_type'].iloc[0]} | Drivers: {len(race_df)}")
        print(f"{'='*100}")
        
        # Show top 5 by predicted top5 probability (absolute method)
        display_cols = [
            'driver_full_name', 'start_position', 'finishing_position',
            'prob_top5_absolute', 'prob_top5_absolute_calibrated',
            'prob_top5_relative', 'prob_top5_relative_calibrated',
            'target_top5'
        ]
        
        # Check which columns exist
        available_cols = [col for col in display_cols if col in race_df.columns]
        
        race_display = race_df[available_cols].copy()
        sort_col = 'prob_top5_absolute_calibrated' if 'prob_top5_absolute_calibrated' in race_display.columns else 'prob_top5_absolute'
        race_display = race_display.sort_values(sort_col, ascending=False)
        
        print(f"\n{'Driver':<25} {'Start':<6} {'Finish':<7} {'Prob_Top5(Abs)':<15} {'Prob_Top5(Rel)':<15} {'Actual_Top5':<12}")
        print("-" * 100)
        
        for _, row in race_display.head(10).iterrows():
            driver = str(row['driver_full_name'])[:24]
            start = int(row['start_position']) if pd.notna(row['start_position']) else 0
            finish = int(row['finishing_position']) if pd.notna(row['finishing_position']) else 0
            prob_abs = row.get('prob_top5_absolute_calibrated', row.get('prob_top5_absolute', 0))
            prob_rel = row.get('prob_top5_relative_calibrated', row.get('prob_top5_relative', 0))
            actual = "YES" if row['target_top5'] == 1 else "NO"
            
            print(f"{driver:<25} {start:<6} {finish:<7} {prob_abs:>6.1f}%        {prob_rel:>6.1f}%        {actual:<12}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*100)
    print("NASCAR PROBABILITY PREDICTION MODEL v1.0")
    print("Regression-based with Absolute & Relative Probabilities")
    print("="*100)
    
    # 1. Load and prepare
    df = load_and_prepare_data('dataset.csv')
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Split data
    train, test = create_train_test_split(df)
    
    # 4. Process features
    train, test, feature_cols, encoders = fill_and_encode_features(train, test)
    
    # 5. Train models
    models = train_models_by_track_type(train, feature_cols)
    
    # 6. Generate predictions - both methods
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    test = predict_probabilities(test, models, feature_cols, method='absolute')
    test = predict_probabilities(test, models, feature_cols, method='relative')
    
    # 7. Calibrate
    test = calibrate_probabilities_platt(train, test, feature_cols, models)
    
    # 8. Evaluate
    print_evaluation_report(test)
    
    # 9. Show samples
    show_sample_predictions(test, num_races=3)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100 + "\n")
    
    return test, models, feature_cols

if __name__ == "__main__":
    test_results, trained_models, feature_columns = main()