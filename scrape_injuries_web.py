import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re

def scrape_football_injury_news():
    """
    Scrape injury data from footballinjurynews.com API
    Returns DataFrame with injury information
    """
    url = "https://footballinjurynews.com/api/premier-league-injuries"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print("Fetching injury data from footballinjurynews.com API...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        injuries = []

        for injury in data:
            injuries.append({
                'Player': injury.get('player_name', ''),
                'Team': injury.get('club', ''),
                'Injury': injury.get('injury', ''),
                'ExpectedReturn': injury.get('expected_return', ''),
                'Position': injury.get('position', ''),
                'ScrapedDate': datetime.now().strftime('%Y-%m-%d')
            })

        # Clean and normalize team names to match historical data
        team_name_mapping = {
            # Current Premier League teams (matching API output format)
            'Arsenal FC': 'Arsenal',
            'Aston Villa': 'Aston Villa',
            'AFC Bournemouth': 'Bournemouth',
            'Brentford FC': 'Brentford',
            'Brighton': 'Brighton',
            'Burnley FC': 'Burnley',
            'Chelsea FC': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton FC': 'Everton',
            'Fulham FC': 'Fulham',
            'Liverpool FC': 'Liverpool',
            'Man City': 'Man City',
            'Man United': 'Man United',
            'Newcastle': 'Newcastle',
            'Nott\'m Forest': 'Nott\'m Forest',
            'Tottenham': 'Tottenham',
            'West Ham': 'West Ham',
            'Wolves': 'Wolves',
            # Alternative names that might appear
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'West Ham United': 'West Ham',
            'Wolverhampton Wanderers': 'Wolves',
            'Brighton & Hove Albion': 'Brighton',
            'Nottingham Forest': 'Nott\'m Forest',
            'Sheffield United': 'Sheffield United',
            'Leeds United': 'Leeds',
            'Luton Town': 'Luton',
            'Ipswich Town': 'Ipswich'
        }

        df = pd.DataFrame(injuries)

        if not df.empty:
            df['Team'] = df['Team'].map(team_name_mapping).fillna(df['Team'])
            # Team names are now properly formatted to match historical data

        print(f"Successfully scraped {len(df)} injury records from footballinjurynews.com")
        return df

    except Exception as e:
        print(f"Error scraping footballinjurynews.com: {e}")
        return pd.DataFrame()

def create_injury_features(historical_data, injury_df):
    """
    Add injury features to historical match data
    """
    if injury_df.empty:
        print("No injury data available, adding zero injury counts")
        historical_data['HomeInjuryCount'] = 0
        historical_data['AwayInjuryCount'] = 0
        historical_data['InjuryAdvantage'] = 0
        return historical_data

    # Group injuries by team and count
    injury_counts = injury_df.groupby('Team').size().reset_index(name='InjuryCount')

    # Create team to injury count mapping
    injury_dict = dict(zip(injury_counts['Team'], injury_counts['InjuryCount']))

    # Add injury counts to historical data
    historical_data['HomeInjuryCount'] = historical_data['HomeTeam'].map(injury_dict).fillna(0).astype(int)
    historical_data['AwayInjuryCount'] = historical_data['AwayTeam'].map(injury_dict).fillna(0).astype(int)
    historical_data['InjuryAdvantage'] = historical_data['AwayInjuryCount'] - historical_data['HomeInjuryCount']

    print(f"Added injury features to {len(historical_data)} matches")
    return historical_data

if __name__ == "__main__":
    # Test the scraper
    injuries = scrape_football_injury_news()
    if not injuries.empty:
        print("\nSample of scraped injuries:")
        print(injuries.head(10))
    else:
        print("No injuries scraped")