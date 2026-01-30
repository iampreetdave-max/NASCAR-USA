"""
NASCAR PROBABILITY PREDICTION MODEL v62-MODULAR
Regression-based approach with relative & absolute probabilities
OPTIMIZED FOR PERFORMANCE + MODULARITY

Key Features:
1. Vectorized race-by-race predictions (wintop5 performance)
2. Modular features system (comment/uncomment to enable/disable)
3. Track-specific blending for WINNER & TOP5
4. Platt scaling calibration
5. Numpy arrays for speed
6. Ensemble models stored as tuples for efficiency

PERFORMANCE NOTES:
- Uses wintop5's vectorized architecture (batch predictions per race)
- Model storage as list of tuples for faster iteration
- MAE reporting for model diagnostics
- >= 50 sample minimum validation
- Fallback logic for unknown model types
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
import pickle
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_RATIO = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.50

# Feature weights - start_position gets most weight
FEATURE_WEIGHTS = {
    'start_position': 3.0,
    'prior_avg_finish_position': 1.5,
    'prior_avg_speed': 1.2,
    'track_avg_finish_position': 1.3,
    'form_last_3': 1.4,
    'form_last_5': 1.2,
}

# =============================================================================
# MODULAR FEATURES - COMMENT/UNCOMMENT TO ENABLE/DISABLE
# =============================================================================
# Simply comment out any line below to exclude it from training/testing.
# The system will automatically skip commented features.
# This allows quick experimentation without code changes.

MODULAR_FEATURES = [
    # Recent driver form (weighted recency)
    'form_last_3',
    'form_last_5',
    'form_last_10',
    
    # Track-specific performance
    'track_avg_finish_position',
    'track_avg_speed',
    'track_races_history',
    
    # Team performance
    'team_momentum',
    
    # Manufacturer performance
    'manu_track_type_avg',
    
    # Win rate metrics
    'win_rate_track_type',
    
    # Career-level driver statistics
    # 'avg_finishing_position',
    # 'recent_finishing_position',
    
    # Driver rating
    # 'avg_driver_rating',
    # 'recent_driver_rating',
    
    # Laps led
    # 'avg_laps_led',
    # 'recent_laps_led',
    
    # Best lap speed
    # 'avg_best_lap_speed',
    # 'recent_best_lap_speed',
    
    # Points
    # 'avg_points',
    # 'recent_points',
    
    # Fastest laps
    # 'avg_fastest_laps',
    # 'recent_fastest_laps',
    
    # Pit stop efficiency
    # 'avg_pit_stop_efficiency',
    # 'recent_pit_stop_efficiency',
    
    # Pre-race qualifying/practice data
    'prior_avg_finish_position',
    'prior_avg_speed',
    'start_position',  # CRITICAL: Keep this - dominates predictions
]

# =============================================================================
# BLENDING CONFIGS (from grid search)
# =============================================================================

WINNER_BLEND_RATIOS = {
    'Short Track': 0.3,
    'Road Course': 0.6,
    'Superspeedway': 0.4,
    'Intermediate': 0.4,
}

TOP5_BLEND_RATIOS = {
    'Short Track': 0.55,
    'Road Course': 0.55,
    'Superspeedway': 0.6,
    'Intermediate': 0.5,
}

TOP5_MODEL_CONFIG = {
    'Short Track': {
        'model': 'lightgbm_deep',
        'params': {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.03,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    'Road Course': {
        'model': 'xgboost',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    'Superspeedway': {
        'model': 'random_forest',
        'params': {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
    },
    'Intermediate': {
        'model': 'xgboost',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    }
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
    
    df = df[(df['elapsed_time_seconds'] > 0) & (df['finishing_position'] <= 40)].copy()
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df.sort_values('race_date').reset_index(drop=True)
    print(f"✓ After filtering DNFs: {len(df)} records")
    
    df['target_winner'] = (df['finishing_position'] == 1).astype(int)
    df['target_top3'] = (df['finishing_position'] <= 3).astype(int)
    df['target_top5'] = (df['finishing_position'] <= 5).astype(int)
    df['target_top10'] = (df['finishing_position'] <= 10).astype(int)
    
    return df

def engineer_features(df):
    """Create pre-race features using only chronological prior data"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING (Modular + Chronological)")
    print("="*80)
    
    # Get active features from MODULAR_FEATURES list
    active_features = [f for f in MODULAR_FEATURES if f not in []]
    
    print(f"\nActive features ({len(active_features)}):")
    for feat in active_features:
        print(f"  ✓ {feat}")
    
    # 1. Driver recent form (last 3, 5, 10 races)
    if any(f in active_features for f in ['form_last_3', 'form_last_5', 'form_last_10']):
        print("\n1. Recent driver form...")
        form_3, form_5, form_10 = [], [], []
        
        for idx, row in df.iterrows():
            mask = (df['driver_full_name'] == row['driver_full_name']) & (df.index < idx)
            past = df.loc[mask, 'finishing_position']
            
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
        
        if 'form_last_3' in active_features:
            df['form_last_3'] = form_3
        if 'form_last_5' in active_features:
            df['form_last_5'] = form_5
        if 'form_last_10' in active_features:
            df['form_last_10'] = form_10
    
    # 2. Track-specific performance
    if any(f in active_features for f in ['track_avg_finish_position', 'track_avg_speed', 'track_races_history']):
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
        
        if 'track_avg_finish_position' in active_features:
            df['track_avg_finish_position'] = track_avg_finish
        if 'track_avg_speed' in active_features:
            df['track_avg_speed'] = track_avg_speed
        if 'track_races_history' in active_features:
            df['track_races_history'] = track_count
    
    # 3. Team performance
    if 'team_momentum' in active_features:
        print("3. Team momentum...")
        team_momentum = []
        
        for idx, row in df.iterrows():
            mask = (df['team'] == row['team']) & (df.index < idx)
            past = df.loc[mask, 'finishing_position'].tail(5)
            team_momentum.append(past.mean() if len(past) > 0 else 20.0)
        
        df['team_momentum'] = team_momentum
    
    # 4. Manufacturer at track type
    if 'manu_track_type_avg' in active_features:
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
    if 'win_rate_track_type' in active_features:
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
    
    # Get active numeric features from MODULAR_FEATURES
    active_features = [f for f in MODULAR_FEATURES if f not in []]
    numeric_features = [col for col in active_features if col in train.columns]
    
    categorical_features = ['driver_full_name', 'team', 'track_name', 'track_type', 'manufacturer']
    
    # Fill missing numeric values with median - STORE MEDIANS FOR SAVING
    medians = {}
    for col in numeric_features:
        if col in train.columns:
            median_val = train[col].median()
            medians[col] = median_val
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
    
    # Apply feature weights
    print("\n" + "="*80)
    print("APPLYING FEATURE WEIGHTS")
    print("="*80)
    
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature in train.columns:
            train[feature] = train[feature] * weight
            test[feature] = test[feature] * weight
            print(f"  {feature:.<40} {weight}x weight applied")
    
    print(f"\n✓ Using {len(feature_cols)} features for training")
    print(f"  Numeric: {len([c for c in feature_cols if '_encoded' not in c])}")
    print(f"  Categorical: {len([c for c in feature_cols if '_encoded' in c])}")
    print(f"\n  Active features: {numeric_features}")
    
    return train, test, feature_cols, encoders, medians

# =============================================================================
# MODEL TRAINING - TRACK-TYPE SPECIFIC (OPTIMIZED FOR PERFORMANCE)
# =============================================================================

def train_models_by_track_type(train, feature_cols):
    """Train ensemble models per track type (vectorized, stores as tuples)"""
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE MODELS (Track-Type Specific)")
    print("="*80)
    
    models = {}
    
    for track_type in sorted(train['track_type'].unique()):
        print(f"\n{track_type}:")
        
        track_train = train[train['track_type'] == track_type].copy()
        
        if len(track_train) < 50:
            print(f"  ⚠ Too few samples ({len(track_train)}), skipping")
            continue
        
        # Use numpy arrays for speed (wintop5 approach)
        X = track_train[feature_cols].values
        y = track_train['finishing_position'].values
        
        ensemble_models = []
        
        # XGBoost
        print(f"  Training XGBoost (n_samples={len(track_train)})...", end=" ")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0
        )
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)
        xgb_mae = mean_absolute_error(y, xgb_pred)
        print(f"✓ MAE: {xgb_mae:.2f}")
        ensemble_models.append(('xgboost', xgb_model))
        
        # LightGBM
        print(f"  Training LightGBM (n_samples={len(track_train)})...", end=" ")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1
        )
        lgb_model.fit(X, y)
        lgb_pred = lgb_model.predict(X)
        lgb_mae = mean_absolute_error(y, lgb_pred)
        print(f"✓ MAE: {lgb_mae:.2f}")
        ensemble_models.append(('lightgbm', lgb_model))
        
        # Random Forest
        print(f"  Training RandomForest (n_samples={len(track_train)})...", end=" ")
        rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        rf_mae = mean_absolute_error(y, rf_pred)
        print(f"✓ MAE: {rf_mae:.2f}")
        ensemble_models.append(('random_forest', rf_model))
        
        # Gradient Boosting
        print(f"  Training GradientBoosting (n_samples={len(track_train)})...", end=" ")
        gb_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE
        )
        gb_model.fit(X, y)
        gb_pred = gb_model.predict(X)
        gb_mae = mean_absolute_error(y, gb_pred)
        print(f"✓ MAE: {gb_mae:.2f}")
        ensemble_models.append(('gradient_boosting', gb_model))
        
        models[track_type] = ensemble_models
        print(f"  ✓ 4-model ensemble trained for {track_type}")
    
    return models

def train_top5_models(train, feature_cols):
    """Train track-specific Top5 models (optimized per track type)"""
    print("\n" + "="*80)
    print("TRAINING TOP5-SPECIFIC MODELS (Track-Optimized)")
    print("="*80)
    
    top5_models = {}
    
    for track_type in sorted(train['track_type'].unique()):
        config = TOP5_MODEL_CONFIG.get(track_type, TOP5_MODEL_CONFIG['Intermediate'])
        model_type = config['model']
        params = config['params']
        
        print(f"\n{track_type} ({model_type}):")
        
        track_train = train[train['track_type'] == track_type].copy()
        
        if len(track_train) < 50:
            print(f"  ⚠ Too few samples ({len(track_train)}), skipping")
            continue
        
        X = track_train[feature_cols].values
        y = track_train['finishing_position'].values
        
        if model_type == 'lightgbm_deep':
            model = lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 10),
                learning_rate=params.get('learning_rate', 0.03),
                num_leaves=params.get('num_leaves', 63),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=RANDOM_STATE, verbose=-1
            )
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 300),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=RANDOM_STATE, verbosity=0
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 300),
                max_depth=params.get('max_depth', 8),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            # Fallback to XGBoost
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=RANDOM_STATE, verbosity=0
            )
        
        model.fit(X, y)
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        print(f"  ✓ {model_type} MAE: {mae:.2f} positions (n={len(track_train)})")
        
        top5_models[track_type] = model
    
    return top5_models

# =============================================================================
# PREDICTION & PROBABILITY GENERATION (VECTORIZED FOR PERFORMANCE)
# =============================================================================

def predict_probabilities(df, models, feature_cols, method='absolute', top5_models=None):
    """
    Generate probabilities (vectorized by race for performance)
    - Batch prediction per race (not row-by-row)
    - Uses numpy arrays for speed
    - Wintop5 architecture maintained
    """
    
    print(f"\n  Generating {method} predictions...", end=" ")
    
    # Initialize columns
    for target in ['winner', 'top3', 'top5', 'top10']:
        df[f'prob_{target}_{method}'] = 0.0
    
    for race_id in df['race_id'].unique():
        race_mask = df['race_id'] == race_id
        race_df = df[race_mask].copy()
        track_type = race_df['track_type'].iloc[0]
        
        if track_type not in models:
            continue
        
        ensemble_models = models[track_type]
        
        # VECTORIZED: Get predictions from all models at once
        X = race_df[feature_cols].values
        all_predictions = []
        
        for model_name, model in ensemble_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Average ensemble predictions
        predicted_positions = np.mean(all_predictions, axis=0)
        predicted_positions = np.clip(predicted_positions, 1, 40)
        
        # Top5 model (if available)
        if top5_models and track_type in top5_models:
            top5_model_predictions = top5_models[track_type].predict(X)
            top5_model_predictions = np.clip(top5_model_predictions, 1, 40)
        else:
            top5_model_predictions = predicted_positions
        
        # BLENDING
        winner_blend_ratio = WINNER_BLEND_RATIOS.get(track_type, 0.4)
        raw_start_positions = race_df['start_position'].values / FEATURE_WEIGHTS.get('start_position', 1.0)
        
        winner_blended_positions = (
            winner_blend_ratio * predicted_positions + 
            (1 - winner_blend_ratio) * raw_start_positions
        )
        winner_blended_positions = np.clip(winner_blended_positions, 1, 40)
        
        top5_blend_ratio = TOP5_BLEND_RATIOS.get(track_type, 0.5)
        top5_blended_positions = (
            top5_blend_ratio * top5_model_predictions + 
            (1 - top5_blend_ratio) * raw_start_positions
        )
        top5_blended_positions = np.clip(top5_blended_positions, 1, 40)
        
        if method == 'absolute':
            for idx, (i, row) in enumerate(race_df.iterrows()):
                pos_pred = predicted_positions[idx]
                winner_pos = winner_blended_positions[idx]
                top5_pos = top5_blended_positions[idx]
                
                prob_winner = 1.0 / (1 + np.exp(0.8 * (winner_pos - 1.5)))
                prob_top3 = 1.0 / (1 + np.exp(0.6 * (pos_pred - 3.5)))
                prob_top5 = 1.0 / (1 + np.exp(0.5 * (top5_pos - 5.5)))
                prob_top10 = 1.0 / (1 + np.exp(0.3 * (pos_pred - 10.5)))
                
                df.loc[i, 'prob_winner_absolute'] = prob_winner * 100
                df.loc[i, 'prob_top3_absolute'] = prob_top3 * 100
                df.loc[i, 'prob_top5_absolute'] = prob_top5 * 100
                df.loc[i, 'prob_top10_absolute'] = prob_top10 * 100
        
        elif method == 'relative':
            winner_ranks = pd.Series(winner_blended_positions).rank(method='average').values
            top5_ranks = pd.Series(top5_blended_positions).rank(method='average').values
            model_ranks = pd.Series(predicted_positions).rank(method='average').values
            max_rank = len(race_df)
            
            for idx, (i, row) in enumerate(race_df.iterrows()):
                winner_rank = winner_ranks[idx]
                top5_rank = top5_ranks[idx]
                rank = model_ranks[idx]
                
                prob_winner = 1.0 if winner_rank <= 1 else 0.5 / winner_rank
                prob_top3 = 1.0 if rank <= 3 else 1.0 - (rank - 3) / (max_rank - 3) if max_rank > 3 else 0.5
                prob_top5 = 1.0 if top5_rank <= 5 else 1.0 - (top5_rank - 5) / (max_rank - 5) if max_rank > 5 else 0.5
                prob_top10 = 1.0 if rank <= 10 else 1.0 - (rank - 10) / (max_rank - 10) if max_rank > 10 else 0.5
                
                df.loc[i, 'prob_winner_relative'] = min(prob_winner, 1.0) * 100
                df.loc[i, 'prob_top3_relative'] = min(prob_top3, 1.0) * 100
                df.loc[i, 'prob_top5_relative'] = min(prob_top5, 1.0) * 100
                df.loc[i, 'prob_top10_relative'] = min(prob_top10, 1.0) * 100
    
    print("✓")
    return df

def calibrate_probabilities_platt(train, test, feature_cols, models):
    """Calibrate probabilities using Platt scaling - RETURNS CALIBRATION MODELS"""
    print("\n" + "="*80)
    print("CALIBRATING PROBABILITIES (Platt Scaling)")
    print("="*80)
    
    calibration_models = {}
    
    for method in ['absolute', 'relative']:
        print(f"\n{method.upper()} Method:")
        calibration_models[method] = {}
        
        for target in ['winner', 'top3', 'top5', 'top10']:
            target_col = f'target_{target}'
            prob_col = f'prob_{target}_{method}'
            
            if prob_col not in train.columns or prob_col not in test.columns:
                continue
            
            X_train = train[prob_col].values.reshape(-1, 1) / 100
            y_train = train[target_col].values
            
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_train, y_train)
            
            # Store calibration model
            calibration_models[method][target] = lr
            
            X_test = test[prob_col].values.reshape(-1, 1) / 100
            calibrated = lr.predict_proba(X_test)[:, 1]
            
            test[f'prob_{target}_{method}_calibrated'] = calibrated * 100
    
    return test, calibration_models

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_precision_at_threshold(df, target, method, threshold=THRESHOLD):
    """Evaluate precision: for predicted top-N drivers, how many actually finished top-N?"""
    
    results = []
    
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id].copy()
        
        actual_count = race_df[f'target_{target}'].sum()
        
        if actual_count == 0:
            continue
        
        prob_col = f'prob_{target}_{method}_calibrated'
        if prob_col not in race_df.columns:
            prob_col = f'prob_{target}_{method}'
        
        race_df = race_df.sort_values(prob_col, ascending=False)
        predicted_topn = race_df.head(actual_count)
        
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
        
        display_cols = [
            'driver_full_name', 'start_position', 'finishing_position',
            'prob_top5_absolute', 'prob_top5_absolute_calibrated',
            'prob_top5_relative', 'prob_top5_relative_calibrated',
            'target_top5'
        ]
        
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
# MODEL SAVING FUNCTIONS (NEW)
# =============================================================================

def save_all_models(models, top5_models, calibration_models, encoders, feature_cols, medians, output_dir='nascar_models'):
    """Save all trained models to PKL files"""
    print("\n" + "="*80)
    print("SAVING MODELS TO PKL FILES")
    print("="*80)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}/")
    
    # 1. Save ensemble models (per track type)
    print("\n1. Saving ensemble models...")
    for track_type, ensemble in models.items():
        filename = f"ensemble_{track_type.lower().replace(' ', '_')}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"   ✓ {filename}")
    
    # 2. Save top5 models (per track type)
    print("\n2. Saving top5 models...")
    for track_type, model in top5_models.items():
        filename = f"top5_{track_type.lower().replace(' ', '_')}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✓ {filename}")
    
    # 3. Save calibration models (relative method only)
    print("\n3. Saving calibration models (relative method)...")
    filename = "calibration_models_relative.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(calibration_models['relative'], f)
    print(f"   ✓ {filename}")
    
    # 4. Save encoders
    print("\n4. Saving encoders...")
    filename = "encoders.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"   ✓ {filename}")
    
    # 5. Save feature columns
    print("\n5. Saving feature columns...")
    filename = "feature_cols.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"   ✓ {filename}")
    
    # 6. Save medians
    print("\n6. Saving medians...")
    filename = "medians.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(medians, f)
    print(f"   ✓ {filename}")
    
    # 7. Save config (blend ratios, feature weights, etc.)
    print("\n7. Saving configuration...")
    config = {
        'FEATURE_WEIGHTS': FEATURE_WEIGHTS,
        'WINNER_BLEND_RATIOS': WINNER_BLEND_RATIOS,
        'TOP5_BLEND_RATIOS': TOP5_BLEND_RATIOS,
        'TOP5_MODEL_CONFIG': TOP5_MODEL_CONFIG,
        'MODULAR_FEATURES': MODULAR_FEATURES,
        'TEST_RATIO': TEST_RATIO,
        'RANDOM_STATE': RANDOM_STATE,
        'THRESHOLD': THRESHOLD,
    }
    filename = "config.pkl"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
    print(f"   ✓ {filename}")
    
    print(f"\n✓ All models saved to '{output_dir}/' directory")
    
    return output_dir

def generate_feature_requirements(feature_cols, medians, encoders, output_dir='nascar_models'):
    """Generate a JSON file with all required features and example values"""
    print("\n" + "="*80)
    print("GENERATING FEATURE REQUIREMENTS FILE")
    print("="*80)
    
    feature_requirements = {
        "description": "Required features for NASCAR prediction model",
        "total_features": len(feature_cols),
        "feature_order": feature_cols,
        "features": {}
    }
    
    for col in feature_cols:
        if col.endswith('_encoded'):
            # Categorical feature
            base_col = col.replace('_encoded', '')
            if base_col in encoders:
                classes = list(encoders[base_col].classes_[:5])  # First 5 examples
                feature_requirements["features"][col] = {
                    "type": "categorical_encoded",
                    "original_column": base_col,
                    "example_value": 0,
                    "description": f"LabelEncoded value of {base_col}",
                    "sample_classes": classes
                }
        else:
            # Numeric feature
            median_val = medians.get(col, 20.0)
            feature_requirements["features"][col] = {
                "type": "numeric",
                "example_value": round(float(median_val), 4),
                "median_value": round(float(median_val), 4),
                "description": f"Numeric feature (fill missing with {round(float(median_val), 4)})"
            }
    
    # Add note about feature weights
    feature_requirements["feature_weights_applied"] = {
        "note": "These weights are ALREADY APPLIED during training. Apply same weights to prediction data.",
        "weights": FEATURE_WEIGHTS
    }
    
    # Add categorical columns info
    feature_requirements["categorical_columns"] = {
        "columns": list(encoders.keys()),
        "note": "Use the saved encoders.pkl to transform these columns"
    }
    
    # Save to JSON
    filename = "feature_requirements.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(feature_requirements, f, indent=2)
    print(f"✓ Saved {filename}")
    
    # Also print summary
    print("\n" + "-"*80)
    print("FEATURE SUMMARY (with example values):")
    print("-"*80)
    print(f"\n{'Feature':<40} {'Type':<20} {'Example Value':<15}")
    print("="*75)
    
    for col in feature_cols:
        if col.endswith('_encoded'):
            base_col = col.replace('_encoded', '')
            print(f"{col:<40} {'categorical':<20} {0:<15}")
        else:
            val = medians.get(col, 20.0)
            print(f"{col:<40} {'numeric':<20} {round(float(val), 2):<15}")
    
    print("\n" + "-"*80)
    print("FEATURE WEIGHTS TO APPLY:")
    print("-"*80)
    for feat, weight in FEATURE_WEIGHTS.items():
        print(f"  {feat}: multiply by {weight}")
    
    return filepath

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*100)
    print("NASCAR PROBABILITY PREDICTION MODEL v62-MODULAR")
    print("Optimized for Performance + Modularity")
    print("="*100)
    
    df = load_and_prepare_data('dataset_with_features.csv')
    df = engineer_features(df)
    train, test = create_train_test_split(df)
    train, test, feature_cols, encoders, medians = fill_and_encode_features(train, test)
    
    models = train_models_by_track_type(train, feature_cols)
    top5_models = train_top5_models(train, feature_cols)
    
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    train = predict_probabilities(train, models, feature_cols, method='absolute', top5_models=top5_models)
    train = predict_probabilities(train, models, feature_cols, method='relative', top5_models=top5_models)
    
    test = predict_probabilities(test, models, feature_cols, method='absolute', top5_models=top5_models)
    test = predict_probabilities(test, models, feature_cols, method='relative', top5_models=top5_models)
    
    test, calibration_models = calibrate_probabilities_platt(train, test, feature_cols, models)
    
    print_evaluation_report(test)
    show_sample_predictions(test, num_races=3)
    
    # =========================================================================
    # SAVE ALL MODELS (NEW)
    # =========================================================================
    output_dir = save_all_models(
        models=models,
        top5_models=top5_models,
        calibration_models=calibration_models,
        encoders=encoders,
        feature_cols=feature_cols,
        medians=medians,
        output_dir='nascar_models'
    )
    
    generate_feature_requirements(feature_cols, medians, encoders, output_dir)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    
    # Print saved files summary
    print("\n" + "="*100)
    print("SAVED MODEL FILES:")
    print("="*100)
    for f in os.listdir(output_dir):
        filepath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {f:<40} {size_kb:>10.1f} KB")
    print("="*100 + "\n")
    
    return test, models, feature_cols

if __name__ == "__main__":
    test_results, trained_models, feature_columns = main()