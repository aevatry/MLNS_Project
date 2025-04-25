import pandas as pd
import os
from typing import Union

from . import get_data_dir

DATA_DIR = get_data_dir()
DTST_PATH = os.path.join(DATA_DIR, "lastfm-dataset-50.snappy.parquet")

def download_dataset()->None:
    """
    Download the 50 users LastFM dataset to the data directory
    """

    
    dataset_url = "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-50.snappy.parquet"

    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR)

    if not os.path.exists(DTST_PATH):
        print(f"Downloading dataset from {dataset_url} to {DTST_PATH}")
        df = pd.read_parquet(dataset_url)
        df.to_parquet(DTST_PATH, compression='snappy', index=False)

def read_dataset()->Union[pd.DataFrame, None]:
    """
    Read the LastFM dataset from the data directory with the 'user_id', 'artist_id', 'track_id' and 'timestamp' columns.
    The returned DataFrame is filtered to only include rows with non-null values.
    """
    if not os.path.exists(DTST_PATH):
        print(f"Dataset not found at {DTST_PATH}. Please download it first.")
        return None
    df = pd.read_parquet(DTST_PATH)
    df = df[['user_id', 'artist_id', 'track_id', 'timestamp']].dropna()
    return df

if __name__ == "__main__":
    download_dataset()