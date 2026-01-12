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

csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv')

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

if not path.exists(csv_path):
    st.warning(f"No historical data file found at `{csv_path}`. Please add your CSV file to get started.")
    st.stop()

df = pd.read_csv(csv_path, sep='\t')

if st.checkbox("Show Raw Data"):
    st.subheader("Historical Data")
    df = df.sort_values(by=['MatchDate', 'KickoffTime'], ascending=[False, False])
    st.dataframe(df, height=get_dataframe_height(df), use_container_width=True, hide_index=True)

if st.checkbox("Show Upcoming Matches"):
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if not path.exists(upcoming_csv):
        st.warning(f"No upcoming fixtures file found at `{upcoming_csv}`. Please run `python fetch_upcoming_fixtures.py` to get upcoming matches.")
    else:
        upcoming_df = pd.read_csv(upcoming_csv)
        st.subheader("Upcoming Premier League Matches")
        st.write(f"Found {len(upcoming_df)} upcoming matches")
        st.write("*Times shown in Eastern Time (ET)*")
        st.dataframe(upcoming_df, height=get_dataframe_height(upcoming_df), use_container_width=True, hide_index=True)

# Initialize variables
model_trained = False
X_train, X_test, y_train, y_test, model, mae, acc = None, None, None, None, None, None, None

# Check which features user wants to see
show_predictive = st.checkbox("Show Predictive Data", key="show_predictive_data")
show_upcoming = st.checkbox("Show Upcoming Predictions", key="show_upcoming_predictions")

if show_predictive or show_upcoming:

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

@st.cache_data
def calculate_feature_importance(_model, _X_test, _y_test, _feature_names):
    n_runs = 10  # Reduced from 20 for better performance
    importances = np.zeros((n_runs, _X_test.shape[1]))

    for i in range(n_runs):
        result = permutation_importance(_model, _X_test, _y_test, n_repeats=1, random_state=42+i, scoring='accuracy')
        importances[i, :] = result.importances_mean

    mean_importance = (importances.mean(axis=0) * 100)
    std_importance = (importances.std(axis=0) * 100)
    importance_df = pd.DataFrame({
        'Feature': _feature_names,
        'MeanImportance': mean_importance,
        'StdImportance': std_importance
    }).rename(columns={'MeanImportance': 'Mean Importance (%)', 'StdImportance': 'Std Importance (%)'}).sort_values('Mean Importance (%)', ascending=False)
    
    return importance_df

if show_predictive and model_trained:
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error (MAE): **{mae:.3f}**")
    st.write(f"Accuracy: **{acc:.3f}**")

    # --- Monte Carlo Permutation Importance ---
    st.subheader("Monte Carlo Feature Importance (Permutation)")
    
    if st.button("Calculate Feature Importance", key="calc_importance"):
        with st.spinner("Calculating feature importance... This may take a moment."):
            importance_df = calculate_feature_importance(model, X_test, y_test, X_train.columns)
        st.dataframe(importance_df, hide_index=True, height=get_dataframe_height(importance_df))

if show_upcoming and model_trained:

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
    display_df = upcoming_df[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']].copy()
    display_df.columns = ['Match Date', 'Kickoff Time', 'Home Team', 'Away Team', 'Home Win %', 'Draw %', 'Away Win %']
    display_df['Home Win %'] = (display_df['Home Win %'] * 100).round(1)
    display_df['Draw %'] = (display_df['Draw %'] * 100).round(1)
    display_df['Away Win %'] = (display_df['Away Win %'] * 100).round(1)

    st.subheader("Upcoming Match Predictions")
    st.write("*Times shown in Eastern Time (ET)*")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=get_dataframe_height(display_df))

