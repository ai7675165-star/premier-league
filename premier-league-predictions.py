import streamlit as st
import pandas as pd
import numpy as np
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
from team_name_mapping import normalize_team_name

DATA_DIR = 'data_files/'

st.set_page_config(page_title="Pitch Oracle - Premier League Historical Data", layout="wide", page_icon=path.join(DATA_DIR, 'favicon.ico'))

st.image(path.join(DATA_DIR, 'logo.png'), width=250)
st.title("Premier League Predictor")

csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

def calculate_feature_importance(model, X_test, y_test, feature_names, n_repeats=5):
    """
    Calculate permutation feature importance for the model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
    
    Returns:
        pd.DataFrame: Feature importance results
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring='accuracy')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Importance %': result.importances_mean * 100,
        'Std': result.importances_std
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df

if not path.exists(csv_path):
    st.warning(f"No historical data file found at `{csv_path}`. Please add your CSV file to get started.")
    st.stop()

df = pd.read_csv(csv_path, sep='\t')

# Data preparation (run for both predictive tabs)
# --- Data Preparation ---
# Assume columns: HomeTeam, AwayTeam, FullTimeResult, plus features
# Encode target: 0 = HomeWin, 1 = Draw, 2 = AwayWin
target_map = {'H': 0, 'D': 1, 'A': 2}
df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
df['target'] = df['FullTimeResult'].map(target_map)

# Drop columns not useful for modeling or that leak the result
drop_cols = [
    'MatchDate', 'KickoffTime', 'FullTimeResult', 'HomeTeam', 'AwayTeam', 'WinningTeam',
    'HomeWin', 'AwayWin', 'Draw',  'HalfTimeHomeWin', 'HalfTimeAwayWin', 'HalfTimeDraw', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
    'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HomePoints', 'AwayPoints',
    'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeFouls', 'AwayFouls', 
    'HomeCorners', 'AwayCorners', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards'
]
X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
y = df['target']

# Fill NA and encode categoricals
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes
X = X.fillna(X.mean(numeric_only=True))

# Select only numeric columns
X = X.select_dtypes(include=[np.number])

# Clean column names for XGBoost compatibility
X.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X.columns]

# Remove columns that are not 1D or have object dtype
bad_cols = []
for col in X.columns:
    if isinstance(X[col].iloc[0], (pd.Series, pd.DataFrame)) or X[col].dtype == 'O':
        bad_cols.append(col)
if bad_cols:
    # st.warning(f"Removing columns with unsupported types for XGBoost: {bad_cols}")
    X = X.drop(columns=bad_cols)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- XGBoost Model ---
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# --- Predictions & MAE ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

model_trained = True

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upcoming Matches", "Predictive Data", "Upcoming Predictions", "Statistics", "Raw Data"])

with tab1:
    # Load upcoming fixtures
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if not path.exists(upcoming_csv):
        st.warning(f"No upcoming fixtures file found at `{upcoming_csv}`. Please run `python fetch_upcoming_fixtures.py` to get upcoming matches.")
    else:
        upcoming_df = pd.read_csv(upcoming_csv)
        st.subheader("Upcoming Premier League Matches")
        st.write(f"Found {len(upcoming_df)} upcoming matches")
        st.write("*Times shown in Eastern Time (ET)*")
        st.dataframe(upcoming_df, height=get_dataframe_height(upcoming_df), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error (MAE): **{mae:.3f}**")
    st.write(f"Accuracy: **{acc:.3f}**")

    # --- Monte Carlo Permutation Importance ---
    st.subheader("Monte Carlo Feature Importance (Permutation)")
    
    if st.button("Calculate Feature Importance", key="calc_importance"):
        with st.spinner("Calculating feature importance... This may take a moment."):
            importance_df = calculate_feature_importance(model, X_test, y_test, X_train.columns)
        st.dataframe(importance_df, hide_index=True, height=get_dataframe_height(importance_df))

with tab3:
    # Load upcoming fixtures
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if not path.exists(upcoming_csv):
        st.warning(f"No upcoming fixtures file found at `{upcoming_csv}`. Please run fetch_upcoming_fixtures.py to get upcoming matches.")
        st.stop()

    upcoming_df = pd.read_csv(upcoming_csv)
    
    # Normalize team names to match historical data
    upcoming_df['HomeTeam'] = upcoming_df['HomeTeam'].apply(normalize_team_name)
    upcoming_df['AwayTeam'] = upcoming_df['AwayTeam'].apply(normalize_team_name)

    # Load team stats
    all_teams = pd.read_csv(path.join(DATA_DIR, 'all_teams.csv'), sep='\t')

    # Merge home team stats (using their home performance stats)
    home_cols = ['Team', 'TeamId', 'HomeGoalsAve', 'HomeGoalsTotal', 'HomeGoalsHalfAve', 'HomeGoalsHalfTotal', 
                 'HomeShotsAve', 'HomeShotsTotal', 'HomeShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 
                 'HomeGameDifferentialAve', 'HomeFirstToSecondHalfGoalRatioAve']
    upcoming_df = pd.merge(
        upcoming_df,
        all_teams[home_cols],
        left_on='HomeTeam', right_on='Team', how='left'
    )
    upcoming_df.drop(columns=['Team'], inplace=True)
    
    # Merge away team stats (using their away performance stats)
    away_cols = ['Team', 'AwayGoalsAve', 'AwayGoalsTotal', 'AwayGoalsHalfAve', 'AwayGoalsHalfTotal', 
                 'AwayShotsAve', 'AwayShotsTotal', 'AwayShotsOnTargetAve', 'AwayFirstHalfDifferentialAve', 
                 'AwayGameDifferentialAve', 'AwayFirstToSecondHalfGoalRatioAve']
    upcoming_df = pd.merge(
        upcoming_df,
        all_teams[away_cols],
        left_on='AwayTeam', right_on='Team', how='left'
    )
    upcoming_df.drop(columns=['Team'], inplace=True)
    
    # Load and merge referee data if available
    referee_csv = path.join(DATA_DIR, 'scraped_referees_test.csv')
    if path.exists(referee_csv):
        referee_df = pd.read_csv(referee_csv)
        referee_df['HomeTeam'] = referee_df['HomeTeam'].apply(normalize_team_name)
        referee_df['AwayTeam'] = referee_df['AwayTeam'].apply(normalize_team_name)
        
        # Merge referee assignments with upcoming fixtures
        upcoming_df = pd.merge(
            upcoming_df,
            referee_df[['Date', 'HomeTeam', 'AwayTeam', 'Referee']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='left'
        )
        
        # Load historical referee statistics
        historical_referee_stats = df[['Referee', 'RefYellowCardsPerGame', 'RefRedCardsPerGame', 'RefFoulsPerGame', 
                                       'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']].drop_duplicates()
        
        # Merge referee statistics
        upcoming_df = pd.merge(
            upcoming_df,
            historical_referee_stats,
            on='Referee',
            how='left'
        )
        
        # Fill missing referee stats with league averages
        ref_cols = ['RefYellowCardsPerGame', 'RefRedCardsPerGame', 'RefFoulsPerGame', 
                   'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']
        for col in ref_cols:
            if col in upcoming_df.columns:
                upcoming_df[col] = upcoming_df[col].fillna(df[col].mean())
    
    # Prepare features for prediction model
    X_upcoming = upcoming_df.drop(columns=['Date', 'Time', 'HomeTeam', 'AwayTeam'], errors='ignore')
    
    # Fill NA
    X_upcoming = X_upcoming.fillna(X_upcoming.mean(numeric_only=True))
    
    # Select numeric
    X_upcoming = X_upcoming.select_dtypes(include=[np.number])
    
    # Clean column names
    X_upcoming.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X_upcoming.columns]
    
    # Train a simple model using ONLY the features we have for upcoming matches
    # Filter historical data to same features
    available_features = X_upcoming.columns.tolist()
    X_simple = X[available_features]
    
    # Train simple model
    simple_model = XGBClassifier(eval_metric='mlogloss', random_state=42, max_depth=4)
    simple_model.fit(X_simple, y)
    
    # Predict probabilities using simple model
    proba = simple_model.predict_proba(X_upcoming)

    # Add predictions to df
    upcoming_df['HomeWin_Prob'] = proba[:, 0]
    upcoming_df['Draw_Prob'] = proba[:, 1]
    upcoming_df['AwayWin_Prob'] = proba[:, 2]

    # Prepare display dataframe with human-readable columns and percentages
    display_cols = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']
    if 'Referee' in upcoming_df.columns:
        display_cols.insert(4, 'Referee')
    
    display_df = upcoming_df[display_cols].copy()
    display_df.columns = ['Match Date', 'Kickoff Time', 'Home Team', 'Away Team'] + \
                        (['Referee'] if 'Referee' in upcoming_df.columns else []) + \
                        ['Home Win %', 'Draw %', 'Away Win %']
    display_df['Home Win %'] = (display_df['Home Win %'] * 100).round(1)
    display_df['Draw %'] = (display_df['Draw %'] * 100).round(1)
    display_df['Away Win %'] = (display_df['Away Win %'] * 100).round(1)

    st.subheader("Upcoming Match Predictions")
    st.write("*Times shown in Eastern Time (ET)*")
    if 'Referee' in upcoming_df.columns:
        st.write("✅ **Referee data integrated** - Predictions now include referee statistics from historical matches")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=get_dataframe_height(display_df))

with tab4:
    st.subheader("Manager Statistics")
    st.write("Historical manager performance metrics calculated from Premier League matches (2021-2026)")
    
    # Extract manager statistics from historical data
    manager_cols = ['HomeManager', 'HomeManagerWinRate', 'HomeManagerGoalsPerGame', 'HomeManagerDefensiveSolidity', 
                   'HomeManagerAttackingThreat', 'HomeManagerTacticalFlexibility']
    
    # Get unique managers and their stats
    manager_stats_df = df[manager_cols].drop_duplicates(subset=['HomeManager']).sort_values('HomeManagerWinRate', ascending=False)
    
    # Rename columns for better display
    manager_stats_df.columns = ['Manager', 'Win Rate', 'Goals per Game', 'Defensive Solidity', 
                               'Attacking Threat', 'Tactical Flexibility']
    
    # Format percentages
    percentage_cols = ['Win Rate']
    for col in percentage_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = (manager_stats_df[col] * 100).round(1)
    
    # Format decimal columns
    decimal_cols = ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']
    for col in decimal_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = manager_stats_df[col].round(2)
    
    st.write(f"**Total Managers:** {len(manager_stats_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Win Rate**: Historical winning percentage as manager")
    st.write("- **Goals per Game**: Average goals scored per match under their management")
    st.write("- **Defensive Solidity**: Rating of defensive organization (higher = better defense)")
    st.write("- **Attacking Threat**: Rating of offensive capability (higher = more dangerous attack)")
    st.write("- **Tactical Flexibility**: Rating of adaptability to different tactical situations")
    
    st.dataframe(manager_stats_df, use_container_width=True, hide_index=True, height=get_dataframe_height(manager_stats_df))
    
    # Add summary statistics
    st.subheader("Manager Summary Statistics")
    st.write("**League-wide averages across all managers:**")
    
    # Calculate averages for numeric columns
    numeric_cols = ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']
    percentage_cols = ['Win Rate']
    
    # Create summary dataframe
    summary_data = {}
    
    # Numeric columns - calculate mean
    for col in numeric_cols:
        if col in manager_stats_df.columns:
            summary_data[col] = manager_stats_df[col].mean()
    
    # Percentage columns - calculate mean and format as percentage
    for col in percentage_cols:
        if col in manager_stats_df.columns:
            raw_values = manager_stats_df[col] / 100  # Convert back from percentage
            summary_data[col] = raw_values.mean()
    
    # Create summary dataframe
    summary_df = pd.DataFrame([summary_data])
    
    # Format the display
    summary_display = summary_df.copy()
    for col in percentage_cols:
        if col in summary_display.columns:
            summary_display[col] = (summary_display[col] * 100).round(1)
    
    for col in ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(2)
    
    st.dataframe(summary_display, use_container_width=True, hide_index=True)
    
    st.subheader("Referee Statistics")
    st.write("Historical referee performance metrics calculated from Premier League matches (2021-2026)")
    
    # Extract referee statistics from historical data
    referee_cols = ['Referee', 'RefTotalMatches', 'RefYellowCardsPerGame', 'RefRedCardsPerGame', 
                   'RefFoulsPerGame', 'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']
    
    referee_stats_df = df[referee_cols].drop_duplicates().sort_values('RefTotalMatches', ascending=False)
    
    # Rename columns for better display
    referee_stats_df.columns = ['Referee', 'Total Matches', 'Yellow Cards/Game', 'Red Cards/Game', 
                               'Fouls/Game', 'Home Yellow Advantage', 'Home Win Rate', 'Away Win Rate', 'Draw Rate']
    
    # Format percentages
    percentage_cols = ['Home Win Rate', 'Away Win Rate', 'Draw Rate']
    for col in percentage_cols:
        if col in referee_stats_df.columns:
            referee_stats_df[col] = (referee_stats_df[col] * 100).round(1)
    
    # Format decimal columns
    decimal_cols = ['Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']
    for col in decimal_cols:
        if col in referee_stats_df.columns:
            referee_stats_df[col] = referee_stats_df[col].round(2)
    
    st.write(f"**Total Referees:** {len(referee_stats_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Yellow/Red Cards per Game**: Disciplinary strictness indicators")
    st.write("- **Home Yellow Advantage**: Positive values indicate referees give more yellow cards to home teams")
    st.write("- **Win Rates**: Historical outcome percentages when referee officiates")
    
    st.dataframe(referee_stats_df, use_container_width=True, hide_index=True, height=get_dataframe_height(referee_stats_df))
    
    # Add summary statistics
    st.subheader("Summary Statistics")
    st.write("**League-wide averages across all referees:**")
    
    # Calculate averages for numeric columns
    numeric_cols = ['Total Matches', 'Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']
    percentage_cols = ['Home Win Rate', 'Away Win Rate', 'Draw Rate']
    
    # Create summary dataframe
    summary_data = {}
    
    # Numeric columns - calculate mean
    for col in numeric_cols:
        if col in referee_stats_df.columns:
            # Convert back to raw values for percentage columns
            if col in ['Home Win Rate', 'Away Win Rate', 'Draw Rate']:
                raw_values = referee_stats_df[col] / 100  # Convert back from percentage
                summary_data[col] = raw_values.mean()
            else:
                summary_data[col] = referee_stats_df[col].mean()
    
    # Percentage columns - calculate mean and format as percentage
    for col in percentage_cols:
        if col in referee_stats_df.columns:
            raw_values = referee_stats_df[col] / 100  # Convert back from percentage
            summary_data[col] = raw_values.mean()
    
    # Create summary dataframe
    summary_df = pd.DataFrame([summary_data])
    
    # Format the display
    summary_display = summary_df.copy()
    for col in percentage_cols:
        if col in summary_display.columns:
            summary_display[col] = (summary_display[col] * 100).round(1)
    
    for col in ['Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(2)
    
    # Rename columns for display
    summary_display.columns = ['Avg Total Matches', 'Avg Yellow Cards/Game', 'Avg Red Cards/Game', 
                              'Avg Fouls/Game', 'Avg Home Yellow Advantage', 'Avg Home Win %', 
                              'Avg Away Win %', 'Avg Draw %']
    
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    # Manager Statistics Section
    st.subheader("Manager Statistics")
    st.write("Historical manager performance metrics and tactical preferences")

    # Extract manager statistics from historical data
    manager_cols = ['HomeManager', 'HomeManagerWinRate', 'HomeManagerGoalsPerGame', 'HomeManagerDefensiveSolidity',
                   'HomeManagerAttackingThreat', 'HomeManagerTacticalFlexibility']
    manager_stats_df = df[manager_cols].drop_duplicates(subset=['HomeManager']).dropna(subset=['HomeManager'])

    # Rename columns for better display
    manager_stats_df.columns = ['Manager', 'Win Rate', 'Goals/Game', 'Defensive Solidity',
                               'Attacking Threat', 'Tactical Flexibility']

    # Format percentages and decimals
    percentage_cols = ['Win Rate', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']
    for col in percentage_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = (manager_stats_df[col] * 100).round(1)

    decimal_cols = ['Goals/Game']
    for col in decimal_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = manager_stats_df[col].round(2)

    st.write(f"**Total Managers:** {len(manager_stats_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Win Rate**: Historical winning percentage")
    st.write("- **Goals/Game**: Average goals scored per match under this manager")
    st.write("- **Defensive Solidity**: Rating of defensive organization (higher = better)")
    st.write("- **Attacking Threat**: Rating of attacking potency (higher = better)")
    st.write("- **Tactical Flexibility**: Ability to adapt formations and tactics")

    st.dataframe(manager_stats_df, use_container_width=True, hide_index=True, height=get_dataframe_height(manager_stats_df))

with tab5:
    st.subheader("Historical Data")
    df_sorted = df.sort_values(by=['MatchDate', 'KickoffTime'], ascending=[False, False])
    st.dataframe(df_sorted, height=get_dataframe_height(df_sorted), use_container_width=True, hide_index=True)

