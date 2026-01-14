import requests
from bs4 import BeautifulSoup
import pandas as pd
from os import path
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo
from team_name_mapping import normalize_team_name

DATA_DIR = 'data_files/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

all_fixtures = []

# Fetch fixtures from ESPN API
# The API may return current/recent matches, so we filter for upcoming only
try:
    # Try to get fixtures for the next 60 days
    today = datetime.now()
    end_date = today + timedelta(days=60)
    date_range = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date_range}'

    print(f"Fetching fixtures from ESPN API for date range: {date_range}...")
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()

        # Extract events (matches)
        if 'events' in data:
            for event in data['events']:
                if 'competitions' in event and len(event['competitions']) > 0:
                    competition = event['competitions'][0]
                    if 'competitors' in competition and len(competition['competitors']) >= 2:
                        competitors = competition['competitors']

                        # Determine home/away
                        home_team = None
                        away_team = None
                        for comp in competitors:
                            if comp.get('homeAway') == 'home':
                                home_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))
                            elif comp.get('homeAway') == 'away':
                                away_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))

                        # Get match date and status
                        match_date = event.get('date', '')
                        status = event.get('status', {}).get('type', {}).get('name', '')

                        if match_date:
                            # Convert UTC time to Eastern Time
                            utc_time = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                            eastern_time = utc_time.astimezone(ZoneInfo('US/Eastern'))
                            date_str = eastern_time.strftime('%Y-%m-%d')
                            time_str = eastern_time.strftime('%H:%M')

                            # Only include upcoming matches (not completed)
                            if status not in ['STATUS_FINAL', 'STATUS_FULL_TIME'] and home_team and away_team:
                                all_fixtures.append({
                                    'Date': date_str,
                                    'Time': time_str,
                                    'HomeTeam': home_team,
                                    'AwayTeam': away_team,
                                    'Status': status
                                })

        print(f"✓ Found {len(all_fixtures)} fixtures from ESPN API date range")
    else:
        print(f"✗ Date range API returned status code {response.status_code}")

        # Fallback to general scoreboard
        url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard'
        print("Trying general scoreboard endpoint...")
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract events (matches)
            if 'events' in data:
                for event in data['events']:
                    if 'competitions' in event and len(event['competitions']) > 0:
                        competition = event['competitions'][0]
                        if 'competitors' in competition and len(competition['competitors']) >= 2:
                            competitors = competition['competitors']

                            # Determine home/away
                            home_team = None
                            away_team = None
                            for comp in competitors:
                                if comp.get('homeAway') == 'home':
                                    home_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))
                                elif comp.get('homeAway') == 'away':
                                    away_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))

                            # Get match date and status
                            match_date = event.get('date', '')
                            status = event.get('status', {}).get('type', {}).get('name', '')

                            if match_date:
                                # Convert UTC time to Eastern Time
                                utc_time = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                                eastern_time = utc_time.astimezone(ZoneInfo('US/Eastern'))
                                date_str = eastern_time.strftime('%Y-%m-%d')
                                time_str = eastern_time.strftime('%H:%M')

                                # Only include upcoming matches (not completed)
                                if status not in ['STATUS_FINAL', 'STATUS_FULL_TIME'] and home_team and away_team:
                                    all_fixtures.append({
                                        'Date': date_str,
                                        'Time': time_str,
                                        'HomeTeam': home_team,
                                        'AwayTeam': away_team,
                                        'Status': status
                                    })

            print(f"✓ Found {len(all_fixtures)} fixtures from general ESPN API")
        else:
            print(f"✗ General API returned status code {response.status_code}")

except Exception as e:
    print(f"✗ API fetch failed: {e}")

# If no fixtures found via API, try checking multiple dates
if len(all_fixtures) == 0:
    print("\nTrying date-specific endpoints...")
    today = datetime.now()
    for i in range(60):  # Check next 60 days instead of 30
        date_to_check = today + timedelta(days=i)
        date_param = date_to_check.strftime('%Y%m%d')

        try:
            url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date_param}'
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'events' in data and len(data['events']) > 0:
                    print(f"  {date_param}: {len(data['events'])} matches")

                    for event in data['events']:
                        if 'competitions' in event and len(event['competitions']) > 0:
                            competition = event['competitions'][0]
                            if 'competitors' in competition and len(competition['competitors']) >= 2:
                                competitors = competition['competitors']

                                home_team = None
                                away_team = None
                                for comp in competitors:
                                    if comp.get('homeAway') == 'home':
                                        home_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))
                                    elif comp.get('homeAway') == 'away':
                                        away_team = normalize_team_name(comp.get('team', {}).get('displayName', ''))

                                # Get match date and time
                                match_date = event.get('date', '')
                                status = event.get('status', {}).get('type', {}).get('name', '')

                                if match_date:
                                    # Convert UTC time to Eastern Time
                                    utc_time = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                                    eastern_time = utc_time.astimezone(ZoneInfo('US/Eastern'))
                                    date_str = eastern_time.strftime('%Y-%m-%d')
                                    time_str = eastern_time.strftime('%H:%M')
                                else:
                                    date_str = date_to_check.strftime('%Y-%m-%d')
                                    time_str = 'TBD'

                                # Only include upcoming matches (not completed)
                                if status not in ['STATUS_FINAL', 'STATUS_FULL_TIME'] and home_team and away_team:
                                    all_fixtures.append({
                                        'Date': date_str,
                                        'Time': time_str,
                                        'HomeTeam': home_team,
                                        'AwayTeam': away_team,
                                        'Status': status
                                    })
        except Exception as e:
            continue
    
    print(f"\n✓ Total fixtures found: {len(all_fixtures)}")

# Remove duplicates
seen = set()
unique_fixtures = []
for f in all_fixtures:
    key = (f['Date'], f['Time'], f['HomeTeam'], f['AwayTeam'])
    if key not in seen:
        seen.add(key)
        unique_fixtures.append({k: v for k, v in f.items() if k != 'Status'})  # Remove status column

# Create DataFrame
df = pd.DataFrame(unique_fixtures)

if len(df) > 0:
    # Check if existing file exists and merge data
    output_path = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            print(f"Found existing fixtures file with {len(existing_df)} matches")

            # Convert date columns to datetime for comparison
            df['Date'] = pd.to_datetime(df['Date'])
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])

            # Filter out past matches from existing data
            today = datetime.now().date()
            existing_df = existing_df[existing_df['Date'].dt.date >= today]

            # Merge new and existing data, keeping newer data for same matches
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Remove duplicates based on date, home team, away team
            combined_df = combined_df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')

            # Sort by date
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)

            # Convert back to string format
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')

            df = combined_df
            print(f"Merged with existing data. Total unique fixtures: {len(df)}")

        except Exception as e:
            print(f"Warning: Could not merge with existing data ({e}). Using new data only.")

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved {len(df)} upcoming fixtures to {output_path}")
    print(df.to_string())
else:
    print("\n⚠ No upcoming fixtures found")
