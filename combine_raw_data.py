import pandas as pd
from os import path

DATA_DIR = 'data_files/'

# URLs for Premier League data from football-data.co.uk
urls = [
    'https://www.football-data.co.uk/mmz4281/2122/E0.csv',  # 2021/2022
    'https://www.football-data.co.uk/mmz4281/2223/E0.csv',  # 2022/2023
    'https://www.football-data.co.uk/mmz4281/2324/E0.csv',  # 2023/2024
    'https://www.football-data.co.uk/mmz4281/2425/E0.csv',  # 2024/2025
    'https://www.football-data.co.uk/mmz4281/2526/E0.csv'   # 2025/2026
]

# Load all data from URLs
dataframes = []
for url in urls:
    try:
        df = pd.read_csv(url)
        dataframes.append(df)
        print(f"Loaded data from {url}")
    except Exception as e:
        print(f"Error loading {url}: {e}")

# Concatenate all data into a single DataFrame
historical_data = pd.concat(dataframes, ignore_index=True)

# Save the combined raw data
historical_data.to_csv(path.join(DATA_DIR, 'combined_historical_data.csv'), sep='\t', index=False)

print("Raw data combined and saved to combined_historical_data.csv")