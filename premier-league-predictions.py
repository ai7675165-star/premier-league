import streamlit as st
import pandas as pd
import numpy as np
import os
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
from scipy import stats
from team_name_mapping import normalize_team_name
from generate_pdf_report import generate_statistical_report, generate_quick_report

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
    Calculate permutation feature importance for the model with statistical significance testing.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
    
    Returns:
        pd.DataFrame: Feature importance results with statistical significance
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring='accuracy')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Importance %': result.importances_mean * 100,
        'Std': result.importances_std
    })
    
    # Add statistical significance testing
    # Calculate z-scores and p-values for each feature
    # Null hypothesis: feature importance = 0 (no importance)
    importance_df['Z_Score'] = importance_df['Importance'] / (importance_df['Std'] + 1e-10)  # Add small epsilon to avoid division by zero
    importance_df['P_Value'] = 2 * (1 - stats.norm.cdf(abs(importance_df['Z_Score'])))  # Two-tailed test
    
    # Add significance level categories
    conditions = [
        (importance_df['P_Value'] < 0.001),
        (importance_df['P_Value'] < 0.01),
        (importance_df['P_Value'] < 0.05),
        (importance_df['P_Value'] < 0.10)
    ]
    choices = ['*** (p < 0.001)', '** (p < 0.01)', '* (p < 0.05)', '. (p < 0.10)']
    importance_df['Significance'] = np.select(conditions, choices, default='Not Significant')
    
    # Add confidence intervals (95%)
    confidence_level = 1.96  # 95% confidence
    importance_df['CI_Lower'] = importance_df['Importance'] - confidence_level * importance_df['Std']
    importance_df['CI_Upper'] = importance_df['Importance'] + confidence_level * importance_df['Std']
    
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
    'HomeCorners', 'AwayCorners', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
    'HomeManagerFormation', 'AwayManagerFormation'  # Excluded from model as not statistically relevant
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
        st.dataframe(upcoming_df, height=get_dataframe_height(upcoming_df), width='stretch', hide_index=True)

with tab2:
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error (MAE): **{mae:.3f}**")
    st.write(f"Accuracy: **{acc:.3f}**")

    # --- Monte Carlo Permutation Importance ---
    st.subheader("Monte Carlo Feature Importance (Permutation)")
    
    if st.button("Calculate Feature Importance", key="calc_importance"):
        with st.spinner("Calculating feature importance... This may take a moment."):
            importance_df = calculate_feature_importance(model, X_test, y_test, X_train.columns)
            st.session_state['importance_df'] = importance_df
    
    # Display results if importance_df is available (either just calculated or from session state)
    if 'importance_df' in st.session_state:
        importance_df = st.session_state['importance_df']
        
        # Display statistical significance legend
        st.info("""
        **📊 Statistical Significance Legend:**
        - *** (p < 0.001): Extremely significant
        - ** (p < 0.01): Very significant  
        - * (p < 0.05): Significant
        - . (p < 0.10): Marginally significant
        - Not Significant: No statistical significance
        """)
        
        # Display top 20 features with enhanced formatting
        st.subheader("Top 20 Features by Importance")
        top_20 = importance_df.head(20).copy()
        
        # Format for display
        display_df = top_20[['Feature', 'Importance %', 'Significance', 'Z_Score', 'P_Value']].copy()
        display_df['Importance %'] = display_df['Importance %'].round(3)
        display_df['Z_Score'] = display_df['Z_Score'].round(2)
        display_df['P_Value'] = display_df['P_Value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        
        # Color code based on significance
        def color_significance(val):
            if val == '*** (p < 0.001)':
                return 'background-color: #d4edda; color: #155724'  # Green
            elif val == '** (p < 0.01)':
                return 'background-color: #d1ecf1; color: #0c5460'  # Blue
            elif val == '* (p < 0.05)':
                return 'background-color: #fff3cd; color: #856404'  # Yellow
            elif val == '. (p < 0.10)':
                return 'background-color: #f8d7da; color: #721c24'  # Red
            else:
                return ''
        
        styled_df = display_df.style.apply(lambda x: [color_significance(val) if col == 'Significance' else '' for col, val in x.items()], axis=1)
        st.dataframe(styled_df, hide_index=True, height=get_dataframe_height(display_df))
        
        # Summary statistics
        st.subheader("Statistical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            significant_count = len(importance_df[importance_df['P_Value'] < 0.05])
            st.metric("Statistically Significant Features", f"{significant_count}/{len(importance_df)}", 
                     f"{(significant_count/len(importance_df)*100):.1f}%")
        
        with col2:
            highly_significant = len(importance_df[importance_df['P_Value'] < 0.01])
            st.metric("Highly Significant (p < 0.01)", highly_significant)
        
        with col3:
            top_importance = importance_df['Importance'].max()
            st.metric("Max Importance Score", f"{top_importance:.4f}")
        
        # Feature categories analysis
        st.subheader("Feature Category Analysis")
        
        # Define feature categories
        categories = {
            'Team Performance': ['Points', 'Goals', 'Shots', 'Differential', 'Momentum'],
            'Betting Odds': ['Bet365', 'Pinnacle', 'William'],
            'Manager': ['Manager', 'MGR'],
            'Referee': ['Referee', 'Ref'],
            'Weather': ['Weather', 'Temperature'],
            'Poisson': ['Poisson'],
            'Shooting': ['ShootingEff', 'xG']
        }
        
        category_stats = []
        for cat_name, keywords in categories.items():
            mask = importance_df['Feature'].str.contains('|'.join(keywords), case=False)
            if mask.any():
                cat_importance = importance_df[mask]['Importance'].sum()
                cat_count = mask.sum()
                significant_in_cat = (importance_df[mask]['P_Value'] < 0.05).sum()
                category_stats.append({
                    'Category': cat_name,
                    'Total Importance': cat_importance,
                    'Feature Count': cat_count,
                    'Significant Features': significant_in_cat,
                    'Significance Rate': significant_in_cat / cat_count if cat_count > 0 else 0
                })
        
        if category_stats:
            cat_df = pd.DataFrame(category_stats)
            cat_df = cat_df.sort_values('Total Importance', ascending=False)
            cat_df['Significance Rate'] = (cat_df['Significance Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(cat_df, hide_index=True)
        
        # PDF Export Section
        st.markdown("---")
        st.subheader("📄 Export Statistical Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Full PDF Report", key="full_pdf", help="Generate comprehensive PDF report with charts and detailed analysis"):
                with st.spinner("Generating comprehensive PDF report..."):
                    # Prepare data for PDF generation
                    model_metrics = {
                        'mae': mae,
                        'accuracy': acc,
                        'total_features': len(importance_df),
                        'significant_features': len(importance_df[importance_df['P_Value'] < 0.05]),
                        'significance_rate': len(importance_df[importance_df['P_Value'] < 0.05]) / len(importance_df) * 100,
                        'top_category': cat_df.iloc[0]['Category'] if len(cat_df) > 0 else 'N/A',
                        'top_reliability': float(cat_df.iloc[0]['Significance Rate'].rstrip('%')) if len(cat_df) > 0 else 0
                    }
                    
                    # Convert category stats for PDF
                    pdf_category_stats = []
                    for _, row in cat_df.iterrows():
                        pdf_category_stats.append({
                            'Category': row['Category'],
                            'Features': row['Feature Count'],
                            'Significant': row['Significant Features'],
                            'Reliability': float(row['Significance Rate'].rstrip('%'))
                        })
                    
                    # Generate PDF
                    pdf_path = generate_statistical_report(importance_df, model_metrics, pdf_category_stats)
                    
                    # Read PDF and create download button
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Comprehensive PDF report generated!")
                    st.download_button(
                        label="📥 Download Full Report",
                        data=pdf_bytes,
                        file_name=pdf_path,
                        mime="application/pdf",
                        key="download_full_pdf"
                    )
                    
                    # Clean up file
                    os.remove(pdf_path)
        
        with col2:
            if st.button("📋 Generate Quick Summary PDF", key="quick_pdf", help="Generate concise PDF summary for quick review"):
                with st.spinner("Generating quick PDF summary..."):
                    # Prepare data for quick PDF
                    model_metrics = {
                        'mae': mae,
                        'accuracy': acc,
                        'total_features': len(importance_df),
                        'significant_features': len(importance_df[importance_df['P_Value'] < 0.05]),
                        'significance_rate': len(importance_df[importance_df['P_Value'] < 0.05]) / len(importance_df) * 100
                    }
                    
                    pdf_category_stats = []  # Not needed for quick report
                    
                    # Generate quick PDF
                    pdf_bytes = generate_quick_report(importance_df, model_metrics, pdf_category_stats)
                    
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"premier_league_quick_report_{timestamp}.pdf"
                    
                    st.success("✅ Quick PDF summary generated!")
                    st.download_button(
                        label="📥 Download Quick Summary",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="download_quick_pdf"
                    )
    
    else:
        st.info("Click 'Calculate Feature Importance' to analyze feature significance and enable PDF export.")

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
    st.dataframe(display_df, width='stretch', hide_index=True, height=get_dataframe_height(display_df))

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
    
    st.dataframe(manager_stats_df, width='stretch', hide_index=True, height=get_dataframe_height(manager_stats_df))
    
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
    
    st.dataframe(summary_display, width='stretch', hide_index=True)
    
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
    
    st.dataframe(referee_stats_df, width='stretch', hide_index=True, height=get_dataframe_height(referee_stats_df))
    
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
    
    st.dataframe(summary_display, width='stretch', hide_index=True)

    st.subheader("Formation Statistics")
    st.write("Historical performance analysis by tactical formation (2021-2026)")
    
    # Extract formation statistics from historical data
    formation_cols = ['HomeManagerFormation', 'AwayManagerFormation', 'FullTimeResult']
    
    # Analyze home formations
    home_formation_stats = df.groupby('HomeManagerFormation').agg({
        'FullTimeResult': ['count', lambda x: (x == 'H').mean(), lambda x: (x == 'D').mean(), lambda x: (x == 'A').mean()]
    }).round(4)
    
    home_formation_stats.columns = ['Matches', 'Home_Win_Rate', 'Draw_Rate', 'Away_Win_Rate']
    home_formation_stats = home_formation_stats.sort_values('Home_Win_Rate', ascending=False)
    
    # Analyze away formations
    away_formation_stats = df.groupby('AwayManagerFormation').agg({
        'FullTimeResult': ['count', lambda x: (x == 'H').mean(), lambda x: (x == 'D').mean(), lambda x: (x == 'A').mean()]
    }).round(4)
    
    away_formation_stats.columns = ['Matches', 'Home_Win_Rate', 'Draw_Rate', 'Away_Win_Rate']
    away_formation_stats = away_formation_stats.sort_values('Away_Win_Rate', ascending=False)
    
    # Create combined display
    combined_stats = []
    for formation in home_formation_stats.index:
        if formation in away_formation_stats.index:
            home_stats = home_formation_stats.loc[formation]
            away_stats = away_formation_stats.loc[formation]
            combined_stats.append({
                'Formation': formation,
                'Home Matches': int(home_stats['Matches']),
                'Home Win Rate': home_stats['Home_Win_Rate'],
                'Away Matches': int(away_stats['Matches']),
                'Away Win Rate': away_stats['Away_Win_Rate'],
                'Total Matches': int(home_stats['Matches'] + away_stats['Matches'])
            })
    
    formation_df = pd.DataFrame(combined_stats).sort_values('Total Matches', ascending=False)
    
    # Format percentages
    percentage_cols = ['Home Win Rate', 'Away Win Rate']
    for col in percentage_cols:
        if col in formation_df.columns:
            formation_df[col] = (formation_df[col] * 100).round(1)
    
    st.write(f"**Total Formations Analyzed:** {len(formation_df)}")
    st.write("**Key Insights:**")
    st.write("- **Home Win Rate**: Percentage of matches won when using this formation at home")
    st.write("- **Away Win Rate**: Percentage of matches won when using this formation away")
    st.write("- **Note**: Formations are associated with managers and may correlate with team quality")
    
    st.dataframe(formation_df, width='stretch', hide_index=True, height=get_dataframe_height(formation_df))
    
    # Add formation summary
    st.subheader("Formation Summary")
    st.write("**Formation popularity and performance overview:**")
    
    # Calculate summary statistics
    summary_stats = {
        'Most Popular Formation': formation_df.loc[formation_df['Total Matches'].idxmax(), 'Formation'],
        'Highest Home Win Rate': f"{formation_df.loc[formation_df['Home Win Rate'].idxmax(), 'Formation']} ({formation_df['Home Win Rate'].max()}%)",
        'Highest Away Win Rate': f"{formation_df.loc[formation_df['Away Win Rate'].idxmax(), 'Formation']} ({formation_df['Away Win Rate'].max()}%)",
        'Average Home Win Rate': f"{formation_df['Home Win Rate'].mean():.1f}%",
        'Average Away Win Rate': f"{formation_df['Away Win Rate'].mean():.1f}%"
    }
    
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
    st.dataframe(summary_df, width='stretch', hide_index=True)

with tab5:
    st.subheader("Historical Data")
    df_sorted = df.sort_values(by=['MatchDate', 'KickoffTime'], ascending=[False, False])
    st.dataframe(df_sorted, height=get_dataframe_height(df_sorted), width='stretch', hide_index=True)

