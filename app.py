"""
Streamlit Web App for Football Match Prediction
Enhanced with Team Logos, Better Layout, and Dropdown Logos
"""
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from PIL import Image
import requests
from io import BytesIO
import base64

# Add src to path
sys.path.append('src')
from src.features_engineering import create_match_features


# Page configuration
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced
st.markdown("""
    <style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Match Date Header */
    .match-date {
        text-align: center;
        font-size: 1.8rem;
        color: #ffffff;
        font-weight: 600;
        margin: 0.5rem 0 0.5rem 0;
        padding: 0.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Team Container */
    .team-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .team-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Team Name */
    .team-name {
        text-align: center;
        font-size: clamp(1.2rem, 2vw, 1.8rem);
        font-weight: bold;
        color: #ffffff;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
        line-height: 1.2;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Team Label (HOME/AWAY) */
    .team-label {
        text-align: center;
        font-size: clamp(0.7rem, 1vw, 1rem);
        color: #a0aec0;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* VS Text */
    .vs-text {
        font-size: 12rem;
        font-weight: 900;
        color: #e74c3c;
        text-align: center;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.3);
        letter-spacing: 8px;
        line-height: 1;
        margin-top: 2rem;
    }
    
    /* VS Container for vertical centering */
    .vs-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        min-height: 280px;
    }
    
    /* Team Card Wrapper for responsive layout */
    .team-card-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 280px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .team-card-wrapper:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Team Logo Box */
    .team-logo-box {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        height: 160px;
        margin-bottom: 0.5rem;
    }
    
    .team-logo-box img {
        max-width: 150px;
        max-height: 150px;
        object-fit: contain;
    }
    
    /* Winner Box */
    .winner-box {
        padding: 2rem 2rem;
        border-radius: 15px;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .winner-label {
        font-size: 0.9rem;
        color: #a0aec0;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .winner-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a9eff;
        margin: 0.5rem 0;
    }
    
    .winner-logo {
        margin-top: 1rem;
    }
    
    .winner-logo img {
        width: 100px;
        height: auto;
    }
    
    /* Draw Box */
    .draw-box {
        padding: 3rem 2rem;
        border-radius: 20px;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .draw-label {
        font-size: 1.2rem;
        color: #a0aec0;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .draw-text {
        font-size: 4rem;
        font-weight: bold;
        color: #ffa15a;
        margin: 0.5rem 0;
    }
    
    /* Logo Container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 150px;
        margin: 0;
    }
    
    /* Winner Logo */
    .winner-logo-container {
        text-align: center;
        margin: 2rem 0;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Team Logo in Text */
    .team-with-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# Team Logos Dictionary
TEAM_LOGOS = {
    'Arsenal': 'https://crests.football-data.org/57.png',
    'Aston Villa': 'https://crests.football-data.org/58.png',
    'Bournemouth': 'https://crests.football-data.org/1044.png',
    'Brentford': 'https://crests.football-data.org/402.png',
    'Brighton': 'https://crests.football-data.org/397.png',
    'Burnley': 'https://crests.football-data.org/328.png',
    'Cardiff': 'https://crests.football-data.org/715.png',
    'Chelsea': 'https://crests.football-data.org/61.png',
    'Crystal Palace': 'https://crests.football-data.org/354.png',
    'Everton': 'https://crests.football-data.org/62.png',
    'Fulham': 'https://crests.football-data.org/63.png',
    'Huddersfield': 'https://crests.football-data.org/394.png',
    'Ipswich': 'https://crests.football-data.org/349.png',
    'Leeds': 'https://crests.football-data.org/341.png',
    'Leicester': 'https://crests.football-data.org/338.png',
    'Liverpool': 'https://crests.football-data.org/64.png',
    'Luton': 'https://crests.football-data.org/389.png',
    'Man City': 'https://crests.football-data.org/65.png',
    'Man United': 'https://crests.football-data.org/66.png',
    'Newcastle': 'https://crests.football-data.org/67.png',
    'Norwich': 'https://crests.football-data.org/68.png',
    "Nott'm Forest": 'https://crests.football-data.org/351.png',
    'Sheffield United': 'https://crests.football-data.org/356.png',
    'Southampton': 'https://crests.football-data.org/340.png',
    'Stoke': 'https://crests.football-data.org/70.png',
    'Sunderland': 'https://crests.football-data.org/71.png',
    'Swansea': 'https://crests.football-data.org/72.png',
    'Tottenham': 'https://crests.football-data.org/73.png',
    'Watford': 'https://crests.football-data.org/346.png',
    'West Brom': 'https://crests.football-data.org/74.png',
    'West Ham': 'https://crests.football-data.org/563.png',
    'Wolves': 'https://crests.football-data.org/76.png',
}


@st.cache_data
def get_team_logo(team_name):
    """Get team logo from URL"""
    try:
        logo_url = TEAM_LOGOS.get(team_name)
        
        if logo_url:
            response = requests.get(logo_url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
    except:
        pass
    
    return None


def get_logo_base64(team_name, size=(30, 30)):
    """Convert team logo to base64 for inline display"""
    try:
        logo = get_team_logo(team_name)
        if logo:
            # Resize logo
            logo.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            logo.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except:
        pass
    
    return None


@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    with open('models/xgboost_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names'], model_data['reverse_label_map']


@st.cache_data
def load_historical_data():
    """Load historical match data (cached)"""
    df = pd.read_csv('data/processed/final_dataset_with_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def display_team_card(team_name, label="HOME", logo_size=150):
    """Display team card with logo and name"""
    
    # Get logo as base64 for proper centering
    logo_b64 = get_logo_base64(team_name, size=(logo_size, logo_size))
    
    if logo_b64:
        logo_html = f'<img src="{logo_b64}" style="width: {logo_size}px; height: auto; object-fit: contain;">'
    else:
        logo_html = '<div style="font-size: 4rem;">⚽</div>'
    
    # Render entire card as HTML for proper centering
    st.markdown(
        f'''
        <div class="team-card-wrapper">
            <div class="team-logo-box">
                {logo_html}
            </div>
            <p class="team-name">{team_name}</p>
            <p class="team-label">{label}</p>
        </div>
        ''',
        unsafe_allow_html=True
    )


def get_team_recent_form(df, team_name, n_matches=5):
    """Get recent form for a team"""
    team_matches = df[
        (df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)
    ].sort_values('Date').tail(n_matches)
    
    if len(team_matches) == 0:
        return None
    
    results = []
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            if match['FTR'] == 'H':
                results.append('W')
            elif match['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')
        else:
            if match['FTR'] == 'A':
                results.append('W')
            elif match['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')
    
    return results


def predict_match(model, feature_names, reverse_label_map, 
                  home_team, away_team, match_date, historical_df):
    """Make prediction for a match"""
    
    match_row = pd.Series({
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Date': pd.to_datetime(match_date)
    })
    
    features = create_match_features(historical_df, match_row, n_matches=5, n_h2h=3)
    feature_vector = np.array([[features[name] for name in feature_names]])
    
    prediction_numeric = model.predict(feature_vector)[0]
    prediction_label = reverse_label_map[prediction_numeric]
    prediction_proba = model.predict_proba(feature_vector)[0]
    
    outcome_map = {'A': 'Away Win', 'D': 'Draw', 'H': 'Home Win'}
    
    return {
        'prediction': outcome_map[prediction_label],
        'prediction_label': prediction_label,
        'probabilities': {
            'Away Win': prediction_proba[0] * 100,
            'Draw': prediction_proba[1] * 100,
            'Home Win': prediction_proba[2] * 100
        },
        'features': features
    }


def create_probability_chart(probabilities, home_team, away_team):
    """Create a bar chart for probabilities"""
    
    outcomes = [f'{away_team} Win', 'Draw', f'{home_team} Win']
    probs = [probabilities['Away Win'], probabilities['Draw'], probabilities['Home Win']]
    colors = ['#ef553b', '#ffa15a', '#00cc96']
    
    fig = go.Figure(data=[
        go.Bar(
            x=outcomes,
            y=probs,
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black'),
            marker_color=colors,
            marker_line_color='rgba(0,0,0,0.3)',
            marker_line_width=2,
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🎯 Match Outcome Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=450,
        template='plotly_white',
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_form_comparison(home_form, away_form, home_team, away_team):
    """Create form comparison visualization"""
    
    if home_form is None or away_form is None:
        return None
    
    max_len = max(len(home_form), len(away_form))
    home_form = home_form + [''] * (max_len - len(home_form))
    away_form = away_form + [''] * (max_len - len(away_form))
    
    fig = go.Figure()
    
    color_map = {'W': '#27ae60', 'D': '#f39c12', 'L': '#e74c3c', '': 'lightgray'}
    
    fig.add_trace(go.Bar(
        name=home_team,
        x=[f'Match {i+1}' for i in range(len(home_form))],
        y=[1] * len(home_form),
        marker_color=[color_map[r] for r in home_form],
        text=home_form,
        textposition='inside',
        textfont=dict(size=16, color='white', family='Arial Black'),
        hovertemplate='<b>' + home_team + '</b><br>Result: %{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name=away_team,
        x=[f'Match {i+1}' for i in range(len(away_form))],
        y=[-1] * len(away_form),
        marker_color=[color_map[r] for r in away_form],
        text=away_form,
        textposition='inside',
        textfont=dict(size=16, color='white', family='Arial Black'),
        hovertemplate='<b>' + away_team + '</b><br>Result: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '📊 Recent Form (Last 5 Matches)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        barmode='relative',
        height=350,
        showlegend=True,
        yaxis={'visible': False},
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        )
    )
    
    return fig


def create_feature_radar(features, home_team, away_team):
    """Create radar chart comparing team features"""
    
    categories = ['Form', 'Goals Scored', 'Shot Accuracy', 'Corners']
    
    home_values = [
        min(features['home_form_L5'] / 3 * 100, 100),
        min(features['home_avg_goals_scored_L5'] / 3 * 100, 100),
        features['home_shot_accuracy_L5'] * 100,
        min(features['home_avg_corners_L5'] / 10 * 100, 100)
    ]
    
    away_values = [
        min(features['away_form_L5'] / 3 * 100, 100),
        min(features['away_avg_goals_scored_L5'] / 3 * 100, 100),
        features['away_shot_accuracy_L5'] * 100,
        min(features['away_avg_corners_L5'] / 10 * 100, 100)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=home_team,
        line_color='#00cc96',
        line_width=3,
        fillcolor='rgba(0, 204, 150, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=away_team,
        line_color='#ef553b',
        line_width=3,
        fillcolor='rgba(239, 85, 59, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                tickfont=dict(size=14, family='Arial Black')
            )
        ),
        showlegend=True,
        title={
            'text': '📈 Team Statistics Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        )
    )
    
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ FOOTBALL MATCH PREDICTOR</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Powered by XGBoost Machine Learning • 50.0% Accuracy</p>', 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Load model and data
    try:
        model, feature_names, reverse_label_map = load_model()
        historical_df = load_historical_data()
    except Exception as e:
        st.error(f"❌ Error loading model or data: {e}")
        st.info("Please make sure 'models/xgboost_model.pkl' and 'data/processed/final_dataset_with_features.csv' exist.")
        st.stop()
    
    # Get available teams
    teams = sorted(historical_df['HomeTeam'].unique())
    
    # Sidebar - Input Section
    st.sidebar.markdown("## ⚙️ Match Configuration")
    st.sidebar.markdown("---")
    
    # Home Team Selection with Logo
    st.sidebar.markdown("### 🏠 Home Team")
    home_team = st.sidebar.selectbox(
        "Select home team",
        teams,
        index=0,
        label_visibility="collapsed"
    )
    
    # Show home team logo in sidebar
    home_logo = get_team_logo(home_team)
    if home_logo:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.image(home_logo, width=80)
    
    st.sidebar.markdown("---")
    
    # Away Team Selection with Logo
    st.sidebar.markdown("### ✈️ Away Team")
    away_team = st.sidebar.selectbox(
        "Select away team",
        [t for t in teams if t != home_team],
        index=0,
        label_visibility="collapsed"
    )
    
    # Show away team logo in sidebar
    away_logo = get_team_logo(away_team)
    if away_logo:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.image(away_logo, width=80)
    
    st.sidebar.markdown("---")
    
    # Date Selection
    last_match_date = historical_df['Date'].max()
    default_date = last_match_date + timedelta(days=7)
    
    st.sidebar.markdown("### 📅 Match Date")
    match_date = st.sidebar.date_input(
        "Select match date",
        value=default_date,
        min_value=last_match_date,
        max_value=last_match_date + timedelta(days=365),
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Predict button
    predict_button = st.sidebar.button(
        "🎯 PREDICT MATCH", 
        type="primary", 
        use_container_width=True
    )
    
    if predict_button:
        
        with st.spinner('🔮 Analyzing teams and calculating prediction...'):
            
            # Make prediction
            result = predict_match(
                model, feature_names, reverse_label_map,
                home_team, away_team, match_date, historical_df
            )
            
            # Match Date Header
            st.markdown(
                f'<p class="match-date">📅 {match_date.strftime("%A, %B %d, %Y")}</p>', 
                unsafe_allow_html=True
            )
            
            # Display teams with logos
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                display_team_card(home_team, "HOME", logo_size=150)
            
            with col2:
                st.markdown(
                    '<div class="vs-container">'
                    '<p class="vs-text">VS</p>'
                    '</div>', 
                    unsafe_allow_html=True
                )
            
            with col3:
                display_team_card(away_team, "AWAY", logo_size=150)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Winner Display with team name and logo
            prediction_label = result['prediction_label']
            
            if prediction_label == 'H':
                # Home Win
                winner_name = home_team
                winner_logo_b64 = get_logo_base64(home_team, size=(100, 100))
                
                logo_html = f'<div class="winner-logo"><img src="{winner_logo_b64}"></div>' if winner_logo_b64 else ''
                
                st.markdown(
                    f'<div class="winner-box">'
                    f'<p class="winner-label">MATCH OUTCOME</p>'
                    f'<p class="winner-name">{winner_name}</p>'
                    f'{logo_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            elif prediction_label == 'A':
                # Away Win
                winner_name = away_team
                winner_logo_b64 = get_logo_base64(away_team, size=(100, 100))
                
                logo_html = f'<div class="winner-logo"><img src="{winner_logo_b64}"></div>' if winner_logo_b64 else ''
                
                st.markdown(
                    f'<div class="winner-box">'
                    f'<p class="winner-label">MATCH OUTCOME</p>'
                    f'<p class="winner-name">{winner_name}</p>'
                    f'{logo_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            else:
                # Draw
                st.markdown(
                    f'<div class="draw-box">'
                    f'<p class="draw-label">MATCH OUTCOME</p>'
                    f'<p class="draw-text">Draw</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Probability Summary Cards
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f'<div style="background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%); '
                    f'padding: 1.5rem; border-radius: 15px; text-align: center; color: white; '
                    f'box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(102, 126, 234, 0.3);">'
                    f'<div style="font-size: 1rem; margin-bottom: 0.5rem; color: #a0aec0;">{away_team} Win</div>'
                    f'<div style="font-size: 2.5rem; font-weight: bold;">{result["probabilities"]["Away Win"]:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div style="background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%); '
                    f'padding: 1.5rem; border-radius: 15px; text-align: center; color: white; '
                    f'box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(102, 126, 234, 0.3);">'
                    f'<div style="font-size: 1rem; margin-bottom: 0.5rem; color: #a0aec0;">Draw</div>'
                    f'<div style="font-size: 2.5rem; font-weight: bold;">{result["probabilities"]["Draw"]:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div style="background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%); '
                    f'padding: 1.5rem; border-radius: 15px; text-align: center; color: white; '
                    f'box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(102, 126, 234, 0.3);">'
                    f'<div style="font-size: 1rem; margin-bottom: 0.5rem; color: #a0aec0;">{home_team} Win</div>'
                    f'<div style="font-size: 2.5rem; font-weight: bold;">{result["probabilities"]["Home Win"]:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Probabilities", 
                "📈 Team Comparison", 
                "🔢 Detailed Stats",
                "📋 Recent Form"
            ])
            
            # Tab 1: Probabilities
            with tab1:
                fig_prob = create_probability_chart(
                    result['probabilities'],
                    home_team,
                    away_team
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                
                st.markdown("### 💡 Interpretation")
                st.info(
                    f"The model predicts **{result['prediction']}** as the most likely outcome. "
                    f"These probabilities are based on recent form, goals, shots, corners, and head-to-head history."
                )
            
            # Tab 2: Team Comparison
            with tab2:
                fig_radar = create_feature_radar(
                    result['features'],
                    home_team,
                    away_team
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 🏠 {home_team}")
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(to right, black, #f0f2f6); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00cc96;'>
                        <p><strong>📊 Form:</strong> {result['features']['home_form_L5']:.2f} pts/match</p>
                        <p><strong>⚽ Goals Scored:</strong> {result['features']['home_avg_goals_scored_L5']:.2f}/match</p>
                        <p><strong>🥅 Goals Conceded:</strong> {result['features']['home_avg_goals_conceded_L5']:.2f}/match</p>
                        <p><strong>🎯 Shot Accuracy:</strong> {result['features']['home_shot_accuracy_L5']*100:.1f}%</p>
                        <p><strong>🚩 Avg Corners:</strong> {result['features']['home_avg_corners_L5']:.1f}/match</p>
                        <p><strong>🤝 H2H Win Rate:</strong> {result['features']['home_h2h_win_rate_L3']*100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(f"### ✈️ {away_team}")
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(to right, black, #f0f2f6); 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef553b;'>
                        <p><strong>📊 Form:</strong> {result['features']['away_form_L5']:.2f} pts/match</p>
                        <p><strong>⚽ Goals Scored:</strong> {result['features']['away_avg_goals_scored_L5']:.2f}/match</p>
                        <p><strong>🥅 Goals Conceded:</strong> {result['features']['away_avg_goals_conceded_L5']:.2f}/match</p>
                        <p><strong>🎯 Shot Accuracy:</strong> {result['features']['away_shot_accuracy_L5']*100:.1f}%</p>
                        <p><strong>🚩 Avg Corners:</strong> {result['features']['away_avg_corners_L5']:.1f}/match</p>
                        <p><strong>🤝 H2H Win Rate:</strong> {result['features']['away_h2h_win_rate_L3']*100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Tab 3: Detailed Stats
            with tab3:
                st.markdown("### 🔢 All Features Used in Prediction")
                
                features_df = pd.DataFrame([
                    {
                        'Feature': name.replace('_', ' ').title().replace('L5', '(Last 5)').replace('L3', '(Last 3)'),
                        'Value': f"{result['features'][name]:.4f}"
                    }
                    for name in feature_names
                ])
                
                st.dataframe(
                    features_df, 
                    use_container_width=True, 
                    hide_index=True,
                    height=600
                )
            
            # Tab 4: Recent Form
            with tab4:
                home_form = get_team_recent_form(historical_df, home_team)
                away_form = get_team_recent_form(historical_df, away_team)
                
                if home_form and away_form:
                    fig_form = create_form_comparison(home_form, away_form, home_team, away_team)
                    st.plotly_chart(fig_form, use_container_width=True)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        form_colors = {'W': '🟢', 'D': '🟡', 'L': '🔴'}
                        form_display = ' '.join([form_colors[r] for r in home_form])
                        st.markdown(
                            f"**{home_team} Last 5 Matches:**  \n{form_display}  \n"
                            f"`{' - '.join(home_form)}`"
                        )
                    
                    with col2:
                        form_display = ' '.join([form_colors[r] for r in away_form])
                        st.markdown(
                            f"**{away_team} Last 5 Matches:**  \n{form_display}  \n"
                            f"`{' - '.join(away_form)}`"
                        )
                    
                    st.markdown("---")
                    st.info("🟢 = Win | 🟡 = Draw | 🔴 = Loss")
                else:
                    st.warning("⚠️ Insufficient recent match data to display form")
    
    else:
        # Welcome Screen
        st.markdown(
            '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
            'padding: 2rem; border-radius: 15px; text-align: center; color: white; '
            'margin: 2rem 0; box-shadow: 0 10px 25px rgba(102,126,234,0.3);">'
            '<h2 style="margin: 0;">👈 Select teams and click "Predict Match"</h2>'
            '<p style="margin-top: 1rem; font-size: 1.1rem;">Get AI-powered predictions based on recent form, statistics, and head-to-head history</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Model Information
        st.markdown("## 📊 Model Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); '
                'padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">'
                '<div style="font-size: 2.5rem; font-weight: bold;">50.0%</div>'
                '<div style="font-size: 0.9rem; margin-top: 0.5rem;">Accuracy</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); '
                'padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">'
                '<div style="font-size: 2.5rem; font-weight: bold;">18</div>'
                '<div style="font-size: 0.9rem; margin-top: 0.5rem;">Features</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #27ae60 0%, #229954 100%); '
                'padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">'
                '<div style="font-size: 2.5rem; font-weight: bold;">2,662</div>'
                '<div style="font-size: 0.9rem; margin-top: 0.5rem;">Training Matches</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); '
                'padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">'
                f'<div style="font-size: 2.5rem; font-weight: bold;">{len(teams)}</div>'
                '<div style="font-size: 0.9rem; margin-top: 0.5rem;">Teams</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Available Teams
        with st.expander("📋 View All Available Teams", expanded=False):
            st.markdown("### Premier League Teams")
            cols = st.columns(4)
            for idx, team in enumerate(teams):
                with cols[idx % 4]:
                    logo = get_team_logo(team)
                    if logo:
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.image(logo, width=30)
                        with col_b:
                            st.write(team)
                    else:
                        st.write(f"⚽ {team}")
        
        # Dataset Overview
        st.markdown("## 📈 Dataset Overview")
        
        total_matches = len(historical_df)
        home_wins = len(historical_df[historical_df['FTR'] == 'H'])
        draws = len(historical_df[historical_df['FTR'] == 'D'])
        away_wins = len(historical_df[historical_df['FTR'] == 'A'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", f"{total_matches:,}")
        
        with col2:
            st.metric("Home Wins", f"{home_wins:,}", f"{home_wins/total_matches*100:.1f}%")
        
        with col3:
            st.metric("Draws", f"{draws:,}", f"{draws/total_matches*100:.1f}%")
        
        with col4:
            st.metric("Away Wins", f"{away_wins:,}", f"{away_wins/total_matches*100:.1f}%")
        
        # Distribution Chart
        st.markdown("### Match Outcome Distribution")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Home Win', 'Draw', 'Away Win'],
                values=[home_wins, draws, away_wins],
                marker_colors=['#00cc96', '#ffa15a', '#ef553b'],
                hole=0.4,
                textfont=dict(size=16, color='white', family='Arial Black')
            )
        ])
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>"
        f"<p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>"
        f"⚽ <strong>Football Match Predictor</strong> | Built with XGBoost & Streamlit"
        f"</p>"
        f"<p style='font-size: 0.9rem; color: #95a5a6;'>"
        f"Data: {historical_df['Date'].min().strftime('%Y')} - {historical_df['Date'].max().strftime('%Y')} | "
        f"Premier League Matches"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
