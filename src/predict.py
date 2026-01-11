"""
Make predictions using the trained model
"""
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from features_engineering import create_match_features


def load_model(model_path='../models/xgboost_model.pkl'):
    """
    Load the trained model
    """
    print("="*60)
    print("LOADING TRAINED MODEL")
    print("="*60)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    label_map = model_data['label_map']
    reverse_label_map = model_data['reverse_label_map']
    
    print(f"✅ Model loaded successfully")
    print(f"✅ Model type: {model_data['model_type']}")
    print(f"✅ Number of features: {model_data['n_features']}")
    
    return model, feature_names, reverse_label_map


def load_historical_data(data_path='../data/processed/final_dataset_with_features.csv'):
    """
    Load historical match data (needed for calculating features)
    """
    print("\n" + "="*60)
    print("LOADING HISTORICAL DATA")
    print("="*60)
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Loaded {len(df)} historical matches")
    print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def predict_match(model, feature_names, reverse_label_map, 
                  home_team, away_team, match_date, historical_df):
    """
    Predict the outcome of a single match
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        reverse_label_map: Map from numeric to string labels (0→'A', 1→'D', 2→'H')
        home_team: Home team name
        away_team: Away team name
        match_date: Date of the match (datetime or string)
        historical_df: DataFrame with historical matches
        
    Returns:
        Dictionary with prediction results
    """
    print("\n" + "="*60)
    print(f"PREDICTING: {home_team} vs {away_team}")
    print("="*60)
    
    # Convert date to datetime if string
    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)
    
    print(f"📅 Match Date: {match_date.strftime('%Y-%m-%d')}")
    
    # Create a dummy match row
    match_row = pd.Series({
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Date': match_date
    })
    
    # Calculate features for this match
    print("\n🔧 Calculating features from historical data...")
    features = create_match_features(historical_df, match_row, n_matches=5, n_h2h=3)
    
    # Check if we have enough historical data
    if features['home_form_L5'] == 0 or features['away_form_L5'] == 0:
        print("\n⚠️  WARNING: Insufficient historical data for one or both teams")
        print("    Predictions may be unreliable")
    
    # Create feature vector in correct order
    feature_vector = np.array([[features[name] for name in feature_names]])
    
    # Make prediction
    prediction_numeric = model.predict(feature_vector)[0]
    prediction_label = reverse_label_map[prediction_numeric]
    
    # Get prediction probabilities
    prediction_proba = model.predict_proba(feature_vector)[0]
    
    # Map to outcome names
    outcome_map = {
        'A': 'Away Win',
        'D': 'Draw',
        'H': 'Home Win'
    }
    
    prediction_outcome = outcome_map[prediction_label]
    
    # Create results dictionary
    results = {
        'home_team': home_team,
        'away_team': away_team,
        'date': match_date,
        'prediction': prediction_outcome,
        'prediction_label': prediction_label,
        'probabilities': {
            'Away Win': prediction_proba[0] * 100,
            'Draw': prediction_proba[1] * 100,
            'Home Win': prediction_proba[2] * 100
        },
        'features': features
    }
    
    return results


def display_prediction(results):
    """
    Display prediction results in a nice format
    """
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n🏟️  Match: {results['home_team']} vs {results['away_team']}")
    print(f"📅 Date: {results['date'].strftime('%Y-%m-%d')}")
    
    print(f"\n🎯 PREDICTED OUTCOME: {results['prediction']}")
    
    print("\n📊 Probabilities:")
    for outcome, prob in results['probabilities'].items():
        bar_length = int(prob / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"   {outcome:12s} {prob:5.1f}% {bar}")
    
    print("\n📈 Key Features:")
    features = results['features']
    print(f"   Home Form (L5): {features['home_form_L5']:.2f} pts/match")
    print(f"   Away Form (L5): {features['away_form_L5']:.2f} pts/match")
    print(f"   Home Goals (L5): {features['home_avg_goals_scored_L5']:.2f}/match")
    print(f"   Away Goals (L5): {features['away_avg_goals_scored_L5']:.2f}/match")
    print(f"   Home H2H Win Rate: {features['home_h2h_win_rate_L3']*100:.1f}%")
    print(f"   Away H2H Win Rate: {features['away_h2h_win_rate_L3']*100:.1f}%")


def predict_multiple_matches(model, feature_names, reverse_label_map, 
                             matches_df, historical_df):
    """
    Predict outcomes for multiple matches
    
    Args:
        model: Trained model
        feature_names: List of feature names
        reverse_label_map: Numeric to string label mapping
        matches_df: DataFrame with columns: HomeTeam, AwayTeam, Date
        historical_df: Historical match data
        
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*60)
    print(f"PREDICTING {len(matches_df)} MATCHES")
    print("="*60)
    
    results = []
    
    for idx, match in matches_df.iterrows():
        result = predict_match(
            model, feature_names, reverse_label_map,
            match['HomeTeam'], match['AwayTeam'], match['Date'],
            historical_df
        )
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'HomeTeam': r['home_team'],
            'AwayTeam': r['away_team'],
            'Date': r['date'],
            'Prediction': r['prediction'],
            'Away_Win_Prob': r['probabilities']['Away Win'],
            'Draw_Prob': r['probabilities']['Draw'],
            'Home_Win_Prob': r['probabilities']['Home Win']
        }
        for r in results
    ])
    
    return results_df


def main():
    """
    Example: Make predictions for upcoming matches
    """
    print("\n" + "="*60)
    print("FOOTBALL MATCH PREDICTION SYSTEM")
    print("="*60)
    
    # Load model and historical data
    model, feature_names, reverse_label_map = load_model()
    historical_df = load_historical_data()
    
    # Example 1: Predict a single match
    print("\n" + "="*60)
    print("EXAMPLE 1: SINGLE MATCH PREDICTION")
    print("="*60)
    
    # Use the most recent date in your dataset + 1 day for prediction
    last_date = historical_df['Date'].max()
    prediction_date = last_date + pd.Timedelta(days=7)
    
    # Predict a match (using teams from your dataset)
    teams = historical_df['HomeTeam'].unique()
    home_team = teams[0]  # First team in dataset
    away_team = teams[1]  # Second team in dataset
    
    results = predict_match(
        model, feature_names, reverse_label_map,
        home_team, away_team, prediction_date,
        historical_df
    )
    
    display_prediction(results)
    
    # Example 2: Predict multiple matches
    print("\n\n" + "="*60)
    print("EXAMPLE 2: MULTIPLE MATCH PREDICTIONS")
    print("="*60)
    
    # Create some example matches
    upcoming_matches = pd.DataFrame({
        'HomeTeam': [teams[0], teams[2], teams[4]],
        'AwayTeam': [teams[1], teams[3], teams[5]],
        'Date': [prediction_date] * 3
    })
    
    predictions_df = predict_multiple_matches(
        model, feature_names, reverse_label_map,
        upcoming_matches, historical_df
    )
    
    print("\n📋 Predictions Summary:")
    print(predictions_df.to_string(index=False))
    
    # Save predictions to CSV
    output_path = '../predictions/predictions.csv'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Predictions saved to: {output_path}")
    
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)
    print("\nNext step: Run the Streamlit app!")
    print("Command: streamlit run app.py")


if __name__ == "__main__":
    main()