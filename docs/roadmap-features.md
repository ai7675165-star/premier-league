# Feature Roadmap - Premier League Predictor

## High Priority Features

### 1. Live Score Integration
**Priority:** High  
**Effort:** Medium  
**Impact:** High

Fetch and display live match scores during game time.

```python
# Add to fetch_upcoming_fixtures.py
def fetch_live_scores():
    """Fetch current in-progress matches with live scores"""
    url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard'
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        live_matches = []
        
        for event in data['events']:
            status = event.get('status', {}).get('type', {}).get('name', '')
            if status in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']:
                competition = event['competitions'][0]
                competitors = competition['competitors']
                
                home = next(c for c in competitors if c['homeAway'] == 'home')
                away = next(c for c in competitors if c['homeAway'] == 'away')
                
                live_matches.append({
                    'HomeTeam': home['team']['displayName'],
                    'AwayTeam': away['team']['displayName'],
                    'HomeScore': home['score'],
                    'AwayScore': away['score'],
                    'Status': status,
                    'Clock': event['status'].get('displayClock', '')
                })
        
        return pd.DataFrame(live_matches)
```

**UI Integration:**
```python
# In premier-league-predictions.py
if st.checkbox("Show Live Matches"):
    st.subheader("Live Premier League Matches")
    live_df = fetch_live_scores()
    
    if len(live_df) > 0:
        st.dataframe(live_df, hide_index=True)
    else:
        st.info("No matches currently in progress")
```

---

### 2. Match Result Confidence Levels
**Priority:** High  
**Effort:** Low  
**Impact:** Medium

Add confidence indicators for predictions based on probability spread.

```python
# Add to premier-league-predictions.py after predictions
def calculate_confidence(proba_row):
    """Calculate prediction confidence level"""
    max_prob = max(proba_row)
    second_prob = sorted(proba_row)[-2]
    spread = max_prob - second_prob
    
    if spread > 0.3:
        return "High"
    elif spread > 0.15:
        return "Medium"
    else:
        return "Low"

# In Show Upcoming Predictions section
upcoming_df['Confidence'] = upcoming_df[['HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']].apply(
    lambda row: calculate_confidence(row.values), axis=1
)

# Display with confidence
st.dataframe(upcoming_df[[
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 
    'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob', 'Confidence'
]], width=900, hide_index=True)
```

---

### 3. Team Form Tracker
**Priority:** Medium  
**Effort:** Medium  
**Impact:** High

Visual display of team form (last 5-10 matches).

```python
# Create new file: analyze_team_form.py
import pandas as pd
from os import path

DATA_DIR = 'data_files/'

def get_team_form(team_name, num_matches=5):
    """Get recent form for a specific team"""
    df = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv'), sep='\t')
    
    # Get matches where team played
    team_matches = df[
        (df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)
    ].sort_values('MatchDate', ascending=False).head(num_matches)
    
    form = []
    for _, match in team_matches.iterrows():
        if match['HomeTeam'] == team_name:
            result = match['FullTimeResult']
            if result == 'H':
                form.append('W')
            elif result == 'D':
                form.append('D')
            else:
                form.append('L')
        else:
            result = match['FullTimeResult']
            if result == 'A':
                form.append('W')
            elif result == 'D':
                form.append('D')
            else:
                form.append('L')
    
    return ''.join(form)

# UI Integration
if st.checkbox("Show Team Form"):
    st.subheader("Team Form Guide")
    teams = sorted(df['HomeTeam'].unique())
    
    form_data = []
    for team in teams:
        form = get_team_form(team)
        wins = form.count('W')
        form_data.append({
            'Team': team,
            'Last 5': form,
            'Wins': wins,
            'Form Score': wins * 3 + form.count('D')
        })
    
    form_df = pd.DataFrame(form_data).sort_values('Form Score', ascending=False)
    st.dataframe(form_df, hide_index=True)
```

---

### 4. Head-to-Head History
**Priority:** Medium  
**Effort:** Low  
**Impact:** Medium

Show historical results between two teams.

```python
# Add to premier-league-predictions.py
def get_h2h_stats(home_team, away_team, num_matches=10):
    """Get head-to-head statistics"""
    df = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv'), sep='\t')
    
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('MatchDate', ascending=False).head(num_matches)
    
    return h2h[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 'FullTimeResult']]

# UI Component
st.subheader("Head-to-Head Analyzer")
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Home Team", sorted(df['HomeTeam'].unique()))
with col2:
    team2 = st.selectbox("Away Team", sorted(df['AwayTeam'].unique()))

if st.button("Get H2H Stats"):
    h2h_df = get_h2h_stats(team1, team2)
    st.dataframe(h2h_df, hide_index=True)
```

---

### 5. Prediction Performance Tracker
**Priority:** Medium  
**Effort:** Medium  
**Impact:** High

Track and display model prediction accuracy over time.

```python
# Create: track_predictions.py
import pandas as pd
from datetime import datetime
from os import path

PREDICTIONS_LOG = 'data_files/predictions_log.csv'

def log_prediction(date, home_team, away_team, pred_home, pred_draw, pred_away):
    """Log a prediction for future validation"""
    prediction = {
        'PredictionDate': datetime.now().strftime('%Y-%m-%d'),
        'MatchDate': date,
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'PredHomeWin': pred_home,
        'PredDraw': pred_draw,
        'PredAwayWin': pred_away,
        'ActualResult': None,  # To be filled after match
        'Correct': None
    }
    
    if path.exists(PREDICTIONS_LOG):
        df = pd.read_csv(PREDICTIONS_LOG)
        df = pd.concat([df, pd.DataFrame([prediction])], ignore_index=True)
    else:
        df = pd.DataFrame([prediction])
    
    df.to_csv(PREDICTIONS_LOG, index=False)

def validate_predictions():
    """Compare predictions with actual results"""
    if not path.exists(PREDICTIONS_LOG):
        return None
    
    predictions = pd.read_csv(PREDICTIONS_LOG)
    historical = pd.read_csv('data_files/combined_historical_data_with_calculations.csv', sep='\t')
    
    for idx, pred in predictions.iterrows():
        if pd.isna(pred['ActualResult']):
            # Find the actual match result
            match = historical[
                (historical['MatchDate'] == pred['MatchDate']) &
                (historical['HomeTeam'] == pred['HomeTeam']) &
                (historical['AwayTeam'] == pred['AwayTeam'])
            ]
            
            if len(match) > 0:
                actual = match.iloc[0]['FullTimeResult']
                predicted = max(
                    [(pred['PredHomeWin'], 'H'), 
                     (pred['PredDraw'], 'D'), 
                     (pred['PredAwayWin'], 'A')]
                )[1]
                
                predictions.at[idx, 'ActualResult'] = actual
                predictions.at[idx, 'Correct'] = (predicted == actual)
    
    predictions.to_csv(PREDICTIONS_LOG, index=False)
    return predictions

# UI Display
if st.checkbox("Show Prediction Performance"):
    st.subheader("Model Prediction Accuracy")
    perf = validate_predictions()
    
    if perf is not None and len(perf) > 0:
        completed = perf[perf['Correct'].notna()]
        accuracy = completed['Correct'].mean()
        st.metric("Prediction Accuracy", f"{accuracy:.1%}")
        st.dataframe(completed, hide_index=True)
```

---

## Medium Priority Features

### 6. Export Predictions to PDF
Generate downloadable PDF reports with predictions.

### 7. Email Alerts for High-Confidence Predictions
Send notifications for matches with >70% confidence.

### 8. Betting Odds Comparison
Compare model predictions with bookmaker odds.

### 9. Multi-League Support
Extend to La Liga, Bundesliga, Serie A.

### 10. Mobile-Responsive Dashboard
Optimize Streamlit UI for mobile devices.

---

## Implementation Timeline

**Phase 1 (Week 1-2):**
- Live Score Integration
- Match Result Confidence Levels
- Team Form Tracker

**Phase 2 (Week 3-4):**
- Head-to-Head History
- Prediction Performance Tracker

**Phase 3 (Month 2):**
- Export to PDF
- Email Alerts
- Betting Odds Comparison

**Phase 4 (Month 3):**
- Multi-League Support
- Mobile Optimization
