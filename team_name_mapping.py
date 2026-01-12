# Team name mapping between ESPN API and historical data
# ESPN uses full official names, historical data uses abbreviated names

TEAM_NAME_MAP = {
    # ESPN name -> Historical data name
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Wolverhampton Wanderers': 'Wolves',
    'Brighton & Hove Albion': 'Brighton',
    'Nottingham Forest': "Nott'm Forest",
    'AFC Bournemouth': 'Bournemouth',
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham',
    'Tottenham Hotspur': 'Tottenham',
    'Leeds United': 'Leeds',
    # Teams that match already
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'Brentford': 'Brentford',
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Liverpool': 'Liverpool',
    'Sunderland': 'Sunderland',
}

def normalize_team_name(team_name):
    """
    Convert ESPN team name to historical data format
    
    Args:
        team_name: Team name from ESPN API
        
    Returns:
        Normalized team name matching historical data
    """
    return TEAM_NAME_MAP.get(team_name, team_name)
