from datetime import date, timedelta
import datetime
import pandas as pd
from os import path
import os
import numpy as np
from scrape_injuries_web import scrape_football_injury_news as scrape_premier_injuries, create_injury_features
from fetch_weather_data import add_weather_features, add_weather_impact_category

DATA_DIR = 'data_files/'

# Load the combined raw historical data
historical_data = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data.csv'), sep='\t')

column_rename_map = {
    # General Info
    'Div': 'Division',
    'Date': 'MatchDate',
    'Time': 'KickoffTime',
    'HomeTeam': 'HomeTeam',
    'AwayTeam': 'AwayTeam',
    'Season': 'Season',
    'Round': 'Round',
    'Venue': 'Venue',
    'Referee': 'Referee',
    'Attendance': 'Attendance',

    # Full Time Results
    'FTHG': 'FullTimeHomeGoals',
    'FTAG': 'FullTimeAwayGoals',
    'FTR': 'FullTimeResult',

    # Half Time Results
    'HTHG': 'HalfTimeHomeGoals',
    'HTAG': 'HalfTimeAwayGoals',
    'HTR': 'HalfTimeResult',

    # Shots, Corners, Fouls, Cards
    'HS': 'HomeShots',
    'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget',
    'AST': 'AwayShotsOnTarget',
    'HHW': 'HomeHitWoodwork',
    'AHW': 'AwayHitWoodwork',
    'HC': 'HomeCorners',
    'AC': 'AwayCorners',
    'HF': 'HomeFouls',
    'AF': 'AwayFouls',
    'HO': 'HomeOffsides',
    'AO': 'AwayOffsides',
    'HY': 'HomeYellowCards',
    'AY': 'AwayYellowCards',
    'HR': 'HomeRedCards',
    'AR': 'AwayRedCards',
    'HBP': 'HomeBookingPoints',
    'ABP': 'AwayBookingPoints',

    # Betting Odds - Bet365
    'B365H': 'Bet365_HomeWinOdds',
    'B365D': 'Bet365_DrawOdds',
    'B365A': 'Bet365_AwayWinOdds',
    'B365>2.5': 'Bet365_Over2_5GoalsOdds',
    'B365<2.5': 'Bet365_Under2_5GoalsOdds',
    'B365AHH': 'Bet365_AH_HomeOdds',
    'B365AHA': 'Bet365_AH_AwayOdds',
    'B365AH': 'Bet365_AH_Handicap',

    # Betting Odds - Other Bookmakers
    'BSH': 'BlueSquare_HomeWinOdds',
    'BSD': 'BlueSquare_DrawOdds',
    'BSA': 'BlueSquare_AwayWinOdds',
    'BWH': 'BetWin_HomeWinOdds',
    'BWD': 'BetWin_DrawOdds',
    'BWA': 'BetWin_AwayWinOdds',
    'GBH': 'Gamebookers_HomeWinOdds',
    'GBD': 'Gamebookers_DrawOdds',
    'GBA': 'Gamebookers_AwayWinOdds',
    'IWH': 'Interwetten_HomeWinOdds',
    'IWD': 'Interwetten_DrawOdds',
    'IWA': 'Interwetten_AwayWinOdds',
    'LBH': 'Ladbrokes_HomeWinOdds',
    'LBD': 'Ladbrokes_DrawOdds',
    'LBA': 'Ladbrokes_AwayWinOdds',
    'PSH': 'Pinnacle_HomeWinOdds',
    'PSD': 'Pinnacle_DrawOdds',
    'PSA': 'Pinnacle_AwayWinOdds',
    'SBH': 'SportingBet_HomeWinOdds',
    'SBD': 'SportingBet_DrawOdds',
    'SBA': 'SportingBet_AwayWinOdds',
    'SJH': 'StanJames_HomeWinOdds',
    'SJD': 'StanJames_DrawOdds',
    'SJA': 'StanJames_AwayWinOdds',
    'SYH': 'Stanleybet_HomeWinOdds',
    'SYD': 'Stanleybet_DrawOdds',
    'SYA': 'Stanleybet_AwayWinOdds',
    'VCH': 'VCBet_HomeWinOdds',
    'VCD': 'VCBet_DrawOdds',
    'VCA': 'VCBet_AwayWinOdds',
    'WHH': 'WilliamHill_HomeWinOdds',
    'WHD': 'WilliamHill_DrawOdds',
    'WHA': 'WilliamHill_AwayWinOdds',

    # Max/Avg Odds
    'MaxH': 'Max_HomeWinOdds',
    'MaxD': 'Max_DrawOdds',
    'MaxA': 'Max_AwayWinOdds',
    'AvgH': 'Avg_HomeWinOdds',
    'AvgD': 'Avg_DrawOdds',
    'AvgA': 'Avg_AwayWinOdds',

    # Betbrain (Bb) Odds
    'Bb1X2': 'Betbrain_NumBookmakers',
    'BbMxH': 'Betbrain_MaxHomeWinOdds',
    'BbAvH': 'Betbrain_AvgHomeWinOdds',
    'BbMxD': 'Betbrain_MaxDrawOdds',
    'BbAvD': 'Betbrain_AvgDrawOdds',
    'BbMxA': 'Betbrain_MaxAwayWinOdds',
    'BbAvA': 'Betbrain_AvgAwayWinOdds',

    # Over/Under Odds
    'BbOU': 'Betbrain_NumBookmakers_OverUnder',
    'BbMx>2.5': 'Betbrain_MaxOver2_5GoalsOdds',
    'BbAv>2.5': 'Betbrain_AvgOver2_5GoalsOdds',
    'BbMx<2.5': 'Betbrain_MaxUnder2_5GoalsOdds',
    'BbAv<2.5': 'Betbrain_AvgUnder2_5GoalsOdds',

    # Asian Handicap Odds
    'BbAH': 'Betbrain_NumBookmakers_AH',
    'BbAHh': 'Betbrain_HandicapSize_Home',
    'BbMxAHH': 'Betbrain_MaxAH_HomeOdds',
    'BbAvAHH': 'Betbrain_AvgAH_HomeOdds',
    'BbMxAHA': 'Betbrain_MaxAH_AwayOdds',
    'BbAvAHA': 'Betbrain_AvgAH_AwayOdds',

    # Corners, Cards, Booking Points Odds (if present)
    'BbMxHC': 'Betbrain_MaxHomeCornersOdds',
    'BbAvHC': 'Betbrain_AvgHomeCornersOdds',
    'BbMxAC': 'Betbrain_MaxAwayCornersOdds',
    'BbAvAC': 'Betbrain_AvgAwayCornersOdds',
    'BbMxHY': 'Betbrain_MaxHomeYellowOdds',
    'BbAvHY': 'Betbrain_AvgHomeYellowOdds',
    'BbMxAY': 'Betbrain_MaxAwayYellowOdds',
    'BbAvAY': 'Betbrain_AvgAwayYellowOdds',
    'BbMxHR': 'Betbrain_MaxHomeRedOdds',
    'BbAvHR': 'Betbrain_AvgHomeRedOdds',
    'BbMxAR': 'Betbrain_MaxAwayRedOdds',
    'BbAvAR': 'Betbrain_AvgAwayRedOdds',

    # Miscellaneous
    'PSCH': 'Pinnacle_ClosingHomeOdds',
    'PSCD': 'Pinnacle_ClosingDrawOdds',
    'PSCA': 'Pinnacle_ClosingAwayOdds',
    'B365CH': 'Bet365_ClosingHomeOdds',
    'B365CD': 'Bet365_ClosingDrawOdds',
    'B365CA': 'Bet365_ClosingAwayOdds',
}

# Example usage:
historical_data.rename(columns=column_rename_map, inplace=True)
historical_data.dropna(axis=1, how='all', inplace=True)
historical_data.drop(columns=['Division'], inplace=True, errors='ignore')

# Parse MatchDate to datetime (already in YYYY-MM-DD format)
historical_data['MatchDate'] = pd.to_datetime(historical_data['MatchDate'], errors='coerce')

# HomeWin, AwayWin, Draw columns
historical_data['HomeWin'] = (historical_data['FullTimeResult'] == 'H').astype(int)
historical_data['AwayWin'] = (historical_data['FullTimeResult'] == 'A').astype(int)
historical_data['Draw'] = (historical_data['FullTimeResult'] == 'D').astype(int)

# WinningTeam column
historical_data['WinningTeam'] = np.where(
    historical_data['FullTimeResult'] == 'H',
    historical_data['HomeTeam'],
    np.where(
        historical_data['FullTimeResult'] == 'A',
        historical_data['AwayTeam'],
        np.nan
    )
)

# Half-time win columns (optional)
historical_data['HalfTimeHomeWin'] = (historical_data['HalfTimeResult'] == 'H').astype(int)
historical_data['HalfTimeAwayWin'] = (historical_data['HalfTimeResult'] == 'A').astype(int)
historical_data['HalfTimeDraw'] = (historical_data['HalfTimeResult'] == 'D').astype(int)

historical_data.to_csv(path.join(DATA_DIR, 'combined_historical_data.csv'), sep='\t', index=False)

home_teams = historical_data['HomeTeam'].unique().tolist()
# print("Home Teams:")
# print(home_teams[:50])
away_teams = historical_data['AwayTeam'].unique().tolist()
# print("Away Teams:")
# print(away_teams[:50])
all_teams = set(home_teams).union(set(away_teams))
# print("All Teams:")
# print(all_teams)
all_teams = pd.DataFrame(all_teams, columns=['Team'])
# print("All Teams DataFrame:")
# print(all_teams)
all_teams['TeamId'] = (
    all_teams['Team']
    .str.replace("'", "", regex=False)   # Remove apostrophes
    .str.replace(' ', '_', regex=False)  # Replace spaces with underscores
).str.lower()

# Calculate stats and merge into all_teams
def add_team_stat(all_teams, historical_data, group_col, value_col, stat_name, func='mean'):
    stat_df = historical_data.groupby(group_col)[value_col].agg(func).reset_index()
    stat_df = stat_df.rename(columns={group_col: 'Team', value_col: stat_name})
    return all_teams.merge(stat_df, on='Team', how='left')

# Averages and totals for goals
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'FullTimeHomeGoals', 'HomeGoalsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'FullTimeAwayGoals', 'AwayGoalsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'FullTimeHomeGoals', 'HomeGoalsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'FullTimeAwayGoals', 'AwayGoalsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HalfTimeHomeGoals', 'HomeGoalsHalfAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'HalfTimeAwayGoals', 'AwayGoalsHalfAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HalfTimeHomeGoals', 'HomeGoalsHalfTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'HalfTimeAwayGoals', 'AwayGoalsHalfTotal', 'sum')

# Shots and shots on target
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShots', 'HomeShotsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShots', 'AwayShotsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShots', 'HomeShotsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShots', 'AwayShotsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShotsOnTarget', 'HomeShotsOnTargetAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShotsOnTarget', 'AwayShotsOnTargetAve', 'mean')

# Differentials (these are not per-team stats, but you can calculate averages per team if needed)
# If you want per-team average differential as home/away:
home_diff = historical_data.copy()
home_diff['HomeFirstHalfDifferential'] = home_diff['HalfTimeHomeGoals'] - home_diff['HalfTimeAwayGoals']
home_diff['HomeGameDifferential'] = home_diff['FullTimeHomeGoals'] - home_diff['FullTimeAwayGoals']
home_diff['HomeSecondHalfGoals'] = home_diff['FullTimeHomeGoals'] - home_diff['HalfTimeHomeGoals']
home_diff['HomeFirstToSecondHalfGoalRatio'] = home_diff['HomeFirstHalfDifferential'] / home_diff['HomeSecondHalfGoals'].replace(0, np.nan)

away_diff = historical_data.copy()
away_diff['AwayFirstHalfDifferential'] = away_diff['HalfTimeAwayGoals'] - away_diff['HalfTimeHomeGoals']
away_diff['AwayGameDifferential'] = away_diff['FullTimeAwayGoals'] - away_diff['FullTimeHomeGoals']
away_diff['AwaySecondHalfGoals'] = away_diff['FullTimeAwayGoals'] - away_diff['HalfTimeAwayGoals']
away_diff['AwayFirstToSecondHalfGoalRatio'] = away_diff['AwayFirstHalfDifferential'] / away_diff['AwaySecondHalfGoals'].replace(0, np.nan)

all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeFirstHalfDifferential', 'HomeFirstHalfDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayFirstHalfDifferential', 'AwayFirstHalfDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeGameDifferential', 'HomeGameDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayGameDifferential', 'AwayGameDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeFirstToSecondHalfGoalRatio', 'HomeFirstToSecondHalfGoalRatioAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayFirstToSecondHalfGoalRatio', 'AwayFirstToSecondHalfGoalRatioAve', 'mean')


# all_teams['HomeGoalsAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['FullTimeHomeGoals'].mean()
# all_teams['AwayGoalsAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['FullTimeAwayGoals'].mean()
# all_teams['HomeGoalsTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['FullTimeHomeGoals'].sum()
# all_teams['AwayGoalsTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['FullTimeAwayGoals'].sum()
# all_teams['HomeGoalsHalfAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HalfTimeHomeGoals'].mean()
# all_teams['AwayGoalsHalfAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['HalfTimeAwayGoals'].mean()
# all_teams['HomeGoalsHalfTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HalfTimeHomeGoals'].sum()
# all_teams['AwayGoalsHalfTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['HalfTimeAwayGoals'].sum()
# all_teams['HomeShotsAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShots'].mean()
# all_teams['AwayShotsAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShots'].mean()
# all_teams['HomeShotsTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShots'].sum()
# all_teams['AwayShotsTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShots'].sum()
# all_teams['HomeShotsOnTargetAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShotsOnTarget'].mean()
# all_teams['AwayShotsOnTargetAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShotsOnTarget'].mean()
# all_teams['HomeFirstHalfDifferential'] = historical_data['HalfTimeHomeGoals'] - historical_data['HalfTimeAwayGoals']
# all_teams['AwayFirstHalfDifferential'] = historical_data['HalfTimeAwayGoals'] - historical_data['HalfTimeHomeGoals']
# all_teams['HomeGameDifferential'] = historical_data['FullTimeHomeGoals'] - historical_data['FullTimeAwayGoals']
# all_teams['AwayGameDifferential'] = historical_data['FullTimeAwayGoals'] - historical_data['FullTimeHomeGoals']

print(all_teams.head(50))

# Calculate points for each match
def get_points(result):
    if result == 'H':
        return 3, 0
    elif result == 'A':
        return 0, 3
    elif result == 'D':
        return 1, 1
    return 0, 0

historical_data['HomePoints'], historical_data['AwayPoints'] = zip(*historical_data['FullTimeResult'].map(get_points))

# Sort by date for rolling calculations
historical_data = historical_data.sort_values('MatchDate')

# Rolling sum of points for last 5 games (recent form)
historical_data['HomeTeamPointsLast5'] = (
    historical_data.groupby('HomeTeam')['HomePoints']
    .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
)
historical_data['AwayTeamPointsLast5'] = (
    historical_data.groupby('AwayTeam')['AwayPoints']
    .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
)

def calc_h2h(row, df, n=5):
    mask = (
        ((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) |
        ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam']))
    )
    prev_matches = df[mask & (df['MatchDate'] < row['MatchDate'])].sort_values('MatchDate', ascending=False).head(n)
    home_wins = ((prev_matches['HomeTeam'] == row['HomeTeam']) & (prev_matches['FullTimeResult'] == 'H')).sum()
    away_wins = ((prev_matches['AwayTeam'] == row['AwayTeam']) & (prev_matches['FullTimeResult'] == 'A')).sum()
    draws = (prev_matches['FullTimeResult'] == 'D').sum()
    return pd.Series([home_wins, away_wins, draws])

historical_data[['HomeH2HWinLast5', 'AwayH2HWinLast5', 'H2HDrawLast5']] = historical_data.apply(
    lambda row: calc_h2h(row, historical_data), axis=1
)

historical_data['Season'] = historical_data['MatchDate'].dt.year  # Adjust if your season format is different

# Cumulative points up to each match
historical_data['HomeTeamCumulativePoints'] = (
    historical_data.groupby(['Season', 'HomeTeam'])['HomePoints'].cumsum() - historical_data['HomePoints']
)
historical_data['AwayTeamCumulativePoints'] = (
    historical_data.groupby(['Season', 'AwayTeam'])['AwayPoints'].cumsum() - historical_data['AwayPoints']
)

historical_data['MatchDate'] = pd.to_datetime(historical_data['MatchDate'], errors='coerce')

def last_match_gap(df, team_col, date_col):
    df = df.sort_values(date_col)
    last_dates = df.groupby(team_col)[date_col].shift(1)
    return (df[date_col] - last_dates).dt.days

historical_data['HomeRestDays'] = last_match_gap(historical_data, 'HomeTeam', 'MatchDate')
historical_data['AwayRestDays'] = last_match_gap(historical_data, 'AwayTeam', 'MatchDate')

# print(all_teams.columns.tolist())

# Save the combined historical data
all_teams.to_csv(path.join(DATA_DIR, 'all_teams.csv'), sep='\t', index=False)
# print(historical_data.head(50))

# 'Team', 'TeamId', 'HomeGoalsAve', 'AwayGoalsAve', 'HomeGoalsTotal', 'AwayGoalsTotal', 'HomeGoalsHalfAve', 'AwayGoalsHalfAve', 'HomeGoalsHalfTotal', 'AwayGoalsHalfTotal', 'HomeShotsAve', 'AwayShotsAve', 'HomeShotsTotal', 'AwayShotsTotal', 'HomeShotsOnTargetAve', 'AwayShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 'AwayFirstHalfDifferentialAve', 'HomeGameDifferentialAve', 'AwayGameDifferentialAve'



historical_data_with_calculations = historical_data.copy()
historical_data_with_calculations = pd.merge(
    historical_data_with_calculations,
    all_teams[['Team', 'TeamId', 'HomeGoalsAve', 'HomeGoalsTotal',  'HomeGoalsHalfAve', 'HomeGoalsHalfTotal', 'HomeShotsAve',  
        'HomeShotsTotal', 'HomeShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 'HomeGameDifferentialAve',  'HomeFirstToSecondHalfGoalRatioAve']],
    left_on='HomeTeam', right_on='Team', how='left', suffixes=('', '_Home')
)
historical_data_with_calculations = pd.merge(
    historical_data_with_calculations,
    all_teams[['Team', 'AwayGoalsAve',  'AwayGoalsTotal',  'AwayGoalsHalfAve', 'AwayGoalsHalfTotal',  
        'AwayShotsAve', 'AwayShotsTotal',  'AwayShotsOnTargetAve', 'AwayFirstHalfDifferentialAve',  'AwayGameDifferentialAve',  'AwayFirstToSecondHalfGoalRatioAve']],
    left_on='AwayTeam', right_on='Team', how='left', suffixes=('', '_Away')
)

# Optionally drop the extra 'Team' columns from the merges
historical_data_with_calculations.drop(columns=['Team', 'Team_Away'], inplace=True, errors='ignore')

def extract_betting_features(df):
    """
    Extract advanced features from betting odds
    Odds data is already available from football-data.co.uk
    """
    print("Extracting betting market features...")

    # Implied probabilities from Bet365 odds
    if 'Bet365_HomeWinOdds' in df.columns:
        # Convert odds to implied probabilities
        df['ImpliedProb_HomeWin'] = 1 / df['Bet365_HomeWinOdds']
        df['ImpliedProb_Draw'] = 1 / df['Bet365_DrawOdds']
        df['ImpliedProb_AwayWin'] = 1 / df['Bet365_AwayWinOdds']

        # Normalize to sum to 1 (remove bookmaker margin)
        total_prob = (df['ImpliedProb_HomeWin'] + df['ImpliedProb_Draw'] + df['ImpliedProb_AwayWin'])
        df['ImpliedProb_HomeWin_Norm'] = df['ImpliedProb_HomeWin'] / total_prob
        df['ImpliedProb_Draw_Norm'] = df['ImpliedProb_Draw'] / total_prob
        df['ImpliedProb_AwayWin_Norm'] = df['ImpliedProb_AwayWin'] / total_prob

        # Market confidence (how much margin the bookmaker is taking)
        df['Bet365_MarketMargin'] = total_prob - 1

        # Odds movement (compare Bet365 vs William Hill if available)
        if 'WilliamHill_HomeWinOdds' in df.columns:
            df['OddsMovement_Home'] = df['Bet365_HomeWinOdds'] - df['WilliamHill_HomeWinOdds']
            df['OddsMovement_Away'] = df['Bet365_AwayWinOdds'] - df['WilliamHill_AwayWinOdds']
            df['OddsMovement_Draw'] = df['Bet365_DrawOdds'] - df['WilliamHill_DrawOdds']
        else:
            df['OddsMovement_Home'] = 0
            df['OddsMovement_Away'] = 0
            df['OddsMovement_Draw'] = 0

        # Odds value indicators (when odds suggest better value than actual probability)
        # Lower odds number = higher probability
        df['Bet365_Value_Home'] = df['ImpliedProb_HomeWin_Norm'] - (1 / df['Bet365_HomeWinOdds'])
        df['Bet365_Value_Away'] = df['ImpliedProb_AwayWin_Norm'] - (1 / df['Bet365_AwayWinOdds'])
        df['Bet365_Value_Draw'] = df['ImpliedProb_Draw_Norm'] - (1 / df['Bet365_DrawOdds'])

        # Odds ratio features
        df['Bet365_HomeVsDraw_Ratio'] = df['Bet365_HomeWinOdds'] / df['Bet365_DrawOdds']
        df['Bet365_AwayVsDraw_Ratio'] = df['Bet365_AwayWinOdds'] / df['Bet365_DrawOdds']
        df['Bet365_HomeVsAway_Ratio'] = df['Bet365_HomeWinOdds'] / df['Bet365_AwayWinOdds']

    # Over/Under odds features
    if 'Bet365_Over2_5GoalsOdds' in df.columns and 'Bet365_Under2_5GoalsOdds' in df.columns:
        df['Bet365_OverUnder_Margin'] = (1 / df['Bet365_Over2_5GoalsOdds']) + (1 / df['Bet365_Under2_5GoalsOdds']) - 1
        df['Bet365_ExpectedTotalGoals'] = (df['Bet365_Over2_5GoalsOdds'] + df['Bet365_Under2_5GoalsOdds']) / 2

    # Asian Handicap features (if available)
    if 'Bet365_AH_HomeOdds' in df.columns and 'Bet365_AH_AwayOdds' in df.columns:
        ah_total = (1 / df['Bet365_AH_HomeOdds']) + (1 / df['Bet365_AH_AwayOdds'])
        df['Bet365_AH_Margin'] = ah_total - 1

    print(f"Added {len([col for col in df.columns if 'Bet365' in col or 'Implied' in col or 'OddsMovement' in col])} betting features")
    return df

# Extract betting features
historical_data_with_calculations = extract_betting_features(historical_data_with_calculations)

# Add injury data
print("Scraping injury data from PremierInjuries.com...")
injury_df = scrape_premier_injuries()
historical_data_with_calculations = create_injury_features(historical_data_with_calculations, injury_df)

# Add weather data
print("Adding weather data from Open-Meteo (completely free)...")
try:
    historical_data_with_calculations = add_weather_features(historical_data_with_calculations)
    historical_data_with_calculations = add_weather_impact_category(historical_data_with_calculations)
    print("Weather data integration completed")
except Exception as e:
    print(f"Warning: Weather data integration failed: {e}")
    print("Continuing without weather data...")

def calculate_advanced_metrics(df):
    """Calculate advanced team performance metrics from HISTORICAL data only"""
    
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    # First, calculate match-level metrics (these will be shifted to create historical averages)
    df['xG_Home_Match'] = (df['HomeShotsOnTarget'] * 0.35 + df['HomeShots'] * 0.10)
    df['xG_Away_Match'] = (df['AwayShotsOnTarget'] * 0.35 + df['AwayShots'] * 0.10)
    df['ShootingEff_Home_Match'] = df['FullTimeHomeGoals'] / (df['HomeShots'] + 0.1)
    df['ShootingEff_Away_Match'] = df['FullTimeAwayGoals'] / (df['AwayShots'] + 0.1)
    df['GoalDiff_Home_Match'] = df['FullTimeHomeGoals'] - df['FullTimeAwayGoals']
    df['GoalDiff_Away_Match'] = df['FullTimeAwayGoals'] - df['FullTimeHomeGoals']
    
    # Now create rolling averages from PAST matches only (using shift to exclude current match)
    # Home team metrics
    df['HomexG_Avg_L5'] = df.groupby('HomeTeam')['xG_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['HomeShootingEff_Avg_L5'] = df.groupby('HomeTeam')['ShootingEff_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['HomeMomentum_L3'] = df.groupby('HomeTeam')['FullTimeHomeGoals'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['HomeGoalDiff_Avg_L5'] = df.groupby('HomeTeam')['GoalDiff_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Away team metrics
    df['AwayxG_Avg_L5'] = df.groupby('AwayTeam')['xG_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayShootingEff_Avg_L5'] = df.groupby('AwayTeam')['ShootingEff_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayMomentum_L3'] = df.groupby('AwayTeam')['FullTimeAwayGoals'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['AwayGoalDiff_Avg_L5'] = df.groupby('AwayTeam')['GoalDiff_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Drop intermediate match-level calculations
    df = df.drop(columns=['xG_Home_Match', 'xG_Away_Match', 'ShootingEff_Home_Match', 
                          'ShootingEff_Away_Match', 'GoalDiff_Home_Match', 'GoalDiff_Away_Match'])
    
    # Fill NaN values for first matches with reasonable defaults
    df['HomexG_Avg_L5'] = df['HomexG_Avg_L5'].fillna(1.5)
    df['AwayxG_Avg_L5'] = df['AwayxG_Avg_L5'].fillna(1.5)
    df['HomeShootingEff_Avg_L5'] = df['HomeShootingEff_Avg_L5'].fillna(0.15)
    df['AwayShootingEff_Avg_L5'] = df['AwayShootingEff_Avg_L5'].fillna(0.15)
    df['HomeMomentum_L3'] = df['HomeMomentum_L3'].fillna(3.0)
    df['AwayMomentum_L3'] = df['AwayMomentum_L3'].fillna(3.0)
    df['HomeGoalDiff_Avg_L5'] = df['HomeGoalDiff_Avg_L5'].fillna(0.0)
    df['AwayGoalDiff_Avg_L5'] = df['AwayGoalDiff_Avg_L5'].fillna(0.0)
    
    return df

# Calculate advanced metrics
print("Calculating advanced team metrics...")
historical_data_with_calculations = calculate_advanced_metrics(historical_data_with_calculations)

# Save the result (all match-level fields are already present)
historical_data_with_calculations.to_csv(
    path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv'),
    sep='\t',
    index=False
)