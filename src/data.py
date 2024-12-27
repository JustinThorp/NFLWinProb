import pandas as pd

data_list = []
for season in range(1999,2025):
    temp_df = pd.read_parquet(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet")
    data_list.append(temp_df)

data = pd.concat(data_list)


data.to_parquet('data/traing_data.parquet')