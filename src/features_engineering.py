import pandas as pd
import numpy as np
from datetime import datetime

def get_team_last_n_matches(df, team_name, date, n=5):
    # Get all matches where this team played (home or away) before the date
    team_matches = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
        (df['Date'] < date)
    ].copy()
    
    # Sort by date and get last N matches
    team_matches = team_matches.sort_values('Date').tail(n)
    
    return team_matches

def get_head_to_head_matches(df, team1, team2, date, n=3):

    # Get all matches between these two teams before the date
    h2h_matches = df[
        (((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
         ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))) &
        (df['Date'] < date)
    ].copy()
    
    # Sort by date and get last N matches
    h2h_matches = h2h_matches.sort_values('Date').tail(n)
    
    return h2h_matches

def h2h_win_rate(h2h_matches, team_name):

    if len(h2h_matches) == 0:
        return 0.5  # Neutral if no history (50%)
    
    points = []
    
    for _, match in h2h_matches.iterrows():
        is_home = match['HomeTeam'] == team_name

        if match['FTR'] == 'D':
            points.append(1)
        elif (is_home and match['FTR'] == 'H') or (not is_home and match['FTR'] == 'A'):
            points.append(3)
        else:
            points.append(0)
    
    total_points = sum(points)
    max_possible_points = len(h2h_matches) * 3
    
    # Win rate as proportion of points earned
    win_rate = total_points / max_possible_points
    
    return win_rate

def team_form(team_matches, team_name):

    if len(team_matches) == 0:
        return 0.0
    
    points = []
    
    for _, match in team_matches.iterrows():
        is_home = match['HomeTeam'] == team_name

        if match['FTR'] == 'D':
            points.append(1)
        elif (is_home and match['FTR'] == 'H') or (not is_home and match['FTR'] == 'A'):
            points.append(3)
        else:
            points.append(0)
    
    return np.mean(points)

def avg_goals_scored(team_matches, team_name):

    if len(team_matches) == 0:
        return 0.0
    
    goals_scored = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_scored.append(match['FTHG'])  # Goals as home team
        else:
            goals_scored.append(match['FTAG'])  # Goals as away team
    
    return np.mean(goals_scored)

def avg_goals_conceded(team_matches, team_name):

    if len(team_matches) == 0:
        return 0.0
    
    goals_conceded = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            goals_conceded.append(match['FTAG'])  # Conceded as home team
        else:
            goals_conceded.append(match['FTHG'])  # Conceded as away team
    
    return np.mean(goals_conceded)

def avg_shots(team_matches, team_name):

    if len(team_matches) == 0:
        return 0.0
    
    shots = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            if pd.notna(match['HS']):
                shots.append(match['HS'])
        else:
            if pd.notna(match['AS']):
                shots.append(match['AS'])
    
    if len(shots) == 0:
        return 0.0
    
    return np.mean(shots)

def avg_shots_on_target(team_matches, team_name):

    if len(team_matches) == 0:
        return 0.0
    
    shots_on_target = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            if pd.notna(match['HST']):
                shots_on_target.append(match['HST'])
        else:
            if pd.notna(match['AST']):
                shots_on_target.append(match['AST'])
    
    if len(shots_on_target) == 0:
        return 0.0
    
    return np.mean(shots_on_target)

def avg_corners(team_matches, team_name):
    """
    🆕 NEW: Calculate average corners in recent matches
    """
    if len(team_matches) == 0:
        return 0.0
    
    corners = []
    
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            if pd.notna(match.get('HC')):  # Use .get() for safety
                corners.append(match['HC'])
        else:
            if pd.notna(match.get('AC')):
                corners.append(match['AC'])
    
    if len(corners) == 0:
        return 0.0
    
    return np.mean(corners)
def shot_accuracy(shots, shots_on_target):

    if shots == 0:
        return 0.0
    
    return shots_on_target / shots

def create_match_features(df, match_row, n_matches=5, n_h2h=3):

    home_team = match_row['HomeTeam']
    away_team = match_row['AwayTeam']
    match_date = match_row['Date']
    
    # Get recent matches for both teams
    home_recent = get_team_last_n_matches(df, home_team, match_date, n_matches)
    away_recent = get_team_last_n_matches(df, away_team, match_date, n_matches)
    
    # Get head-to-head matches
    h2h_matches = get_head_to_head_matches(df, home_team, away_team, match_date, n_h2h)
    
    # Calculate home team features
    home_form = team_form(home_recent, home_team)
    home_avg_goals_scored = avg_goals_scored(home_recent, home_team)
    home_avg_goals_conceded = avg_goals_conceded(home_recent, home_team)
    home_avg_shots = avg_shots(home_recent, home_team)
    home_avg_shots_on_target = avg_shots_on_target(home_recent, home_team)
    home_shot_accuracy = shot_accuracy(home_avg_shots, home_avg_shots_on_target)
    home_avg_corners = avg_corners(home_recent, home_team)
    
    # Calculate away team features
    away_form = team_form(away_recent, away_team)
    away_avg_goals_scored = avg_goals_scored(away_recent, away_team)
    away_avg_goals_conceded = avg_goals_conceded(away_recent, away_team)
    away_avg_shots = avg_shots(away_recent, away_team)
    away_avg_shots_on_target = avg_shots_on_target(away_recent, away_team)
    away_shot_accuracy = shot_accuracy(away_avg_shots, away_avg_shots_on_target)
    away_avg_corners = avg_corners(away_recent, away_team)
    
    # Calculate head-to-head features
    home_h2h_win_rate = h2h_win_rate(h2h_matches, home_team)
    away_h2h_win_rate = h2h_win_rate(h2h_matches, away_team)
    
    # Calculate derived features
    home_goal_diff = home_avg_goals_scored - home_avg_goals_conceded
    away_goal_diff = away_avg_goals_scored - away_avg_goals_conceded
    
    features = {
        'home_form_L5': home_form,
        'home_avg_goals_scored_L5': home_avg_goals_scored,
        'home_avg_goals_conceded_L5': home_avg_goals_conceded,
        'home_goal_diff_L5': home_goal_diff,
        'home_avg_shots_L5': home_avg_shots,
        'home_avg_shots_on_target_L5': home_avg_shots_on_target,
        'home_shot_accuracy_L5': home_shot_accuracy,
        'home_avg_corners_L5': home_avg_corners,
        'home_h2h_win_rate_L3': home_h2h_win_rate,
        
        'away_form_L5': away_form,
        'away_avg_goals_scored_L5': away_avg_goals_scored,
        'away_avg_goals_conceded_L5': away_avg_goals_conceded,
        'away_goal_diff_L5': away_goal_diff,
        'away_avg_shots_L5': away_avg_shots,
        'away_avg_shots_on_target_L5': away_avg_shots_on_target,
        'away_shot_accuracy_L5': away_shot_accuracy,
        'away_avg_corners_L5': away_avg_corners,
        'away_h2h_win_rate_L3': away_h2h_win_rate,
    }
    
    return features

def create_all_features(df, n_matches=5, n_h2h=3):

    print("="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    print(f"Creating features based on:")
    print(f"  - Last {n_matches} matches (general form)")
    print(f"  - Last {n_h2h} head-to-head matches (H2H record)")
    print(f"\nTotal matches to process: {len(df)}")
    print("\nThis may take a few minutes...\n")
    
    # Initialize feature columns
    feature_columns = [
        'home_form_L5', 'home_avg_goals_scored_L5', 'home_avg_goals_conceded_L5',
        'home_goal_diff_L5', 'home_avg_shots_L5', 'home_avg_shots_on_target_L5',
        'home_shot_accuracy_L5','home_avg_corners_L5','home_h2h_win_rate_L3',
        'away_form_L5', 'away_avg_goals_scored_L5', 'away_avg_goals_conceded_L5',
        'away_goal_diff_L5', 'away_avg_shots_L5', 'away_avg_shots_on_target_L5',
        'away_shot_accuracy_L5','away_avg_corners_L5','away_h2h_win_rate_L3',
    ]
    
    # Create empty columns
    for col in feature_columns:
        df[col] = 0.0
    
    # Process each match
    start_time = datetime.now()
    
    for idx in df.index:
        # Progress indicator
        if idx % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (idx / len(df)) * 100
            print(f"Progress: {idx}/{len(df)} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        
        match_row = df.loc[idx]
        
        # Calculate features for this match
        features = create_match_features(df, match_row, n_matches, n_h2h)
        
        # Update the dataframe
        for feature_name, feature_value in features.items():
            df.at[idx, feature_name] = feature_value
    
    print(f"\n✅ Feature engineering complete!")
    print(f"Total time: {(datetime.now() - start_time).total_seconds():.1f} seconds")
    
    return df


def save_features(df, output_path='../data/processed/final_dataset_with_features.csv'):

    import os
    
    print("\n" + "="*60)
    print("SAVING DATASET WITH FEATURES")
    print("="*60)
    
    # Save to CSV
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    print("\n🚀 Starting Feature Engineering Process...\n")
    
    # Load the cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv('../data/processed/final_dataset.csv')
    
    # Convert Date to datetime if needed
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Loaded {len(df)} matches")
    print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Create features
    df_with_features = create_all_features(df, n_matches=5, n_h2h=3)
        
    # Save
    save_features(df_with_features)