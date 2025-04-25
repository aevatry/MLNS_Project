import networkx as nx
import pandas as pd
import numpy as np
import os
from enum import Enum
from typing import Union

from . import get_data_dir

class Paths(Enum):
    """
    Enum for the paths to the data files.
    """
    DATA_DIR = get_data_dir()
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR)
    GRAPH_PATH = os.path.join(DATA_DIR, "lastfm_graph.graphml")

def build_graph_from_dataframe(df:pd.DataFrame)->nx.Graph:
    """
    Build a graph from the pandas DataFrame representation of the LastFM dataset and saves it to the local data folder.
    Only 3 columns are expected and used: 'user_id', 'artist_id', and 'track_id'.

    Args:
        df (pd.DataFrame): DataFrame to construct the graph from. It is expected that the rows have been cleared of NaN values.

    Returns:
        nx.Graph: Graph representation of the LastFM dataset.
    """

    # Create a new graph
    G = nx.Graph()

    #### For the following, could also use the df.dropna() method to remove rows with NaN values
    unique_users_idx = df['user_id'].notnull()
    unique_artists_idx = df['artist_id'].notnull()
    unique_tracks_idx = df['track_id'].notnull()

    unique_users = df[unique_users_idx]['user_id'].unique()
    unique_artists = df[unique_artists_idx]['artist_id'].unique()
    unique_tracks = df[unique_tracks_idx]['track_id'].unique()

    all_nodes = np.concatenate([unique_users, unique_artists, unique_tracks])

    for i, id in enumerate(all_nodes):

        if id is not None:
            if i < len(unique_users):
                node_type = 'user'
            elif i < len(unique_users) + len(unique_artists):
                node_type = 'artist'
            elif i < len(unique_users) + len(unique_artists) + len(unique_tracks):
                node_type = 'track'
            else:
                raise ValueError(f"Node index {i} out of range")
            G.add_node(id, node_type=node_type)

    # Add edges between users and artists

    for group in df.groupby(['user_id', 'artist_id']):
        # group by user and artist such that we create edges faster
        user_id, artist_id = group[0]

        # add the user-artist edge
        if user_id is not None and artist_id is not None:
            G.add_edge(user_id, artist_id)

        tracks = group[1]['track_id'].values
        for track_id in tracks:
            if track_id is not None and user_id is not None:
                G.add_edge(user_id, track_id)
            if track_id is not None and artist_id is not None:
                G.add_edge(artist_id, track_id)
        

    # Save the graph to a GraphML file
    nx.write_graphml(G, Paths.GRAPH_PATH.value)
    print(f"Graph saved to {Paths.GRAPH_PATH.value}")

    return G


def read_graph_from_data_dir()->Union[nx.Graph, None]:
    """
    Read a the LastFM graph from a GraphML file in the data directory.

    Returns:
        nx.Graph: The graph read from the file. If no file found, return None
    """

    try:
        G = nx.read_graphml(Paths.GRAPH_PATH.value)
        print(f"Graph loaded from {Paths.GRAPH_PATH.value}")
        return G
    except FileNotFoundError:
        print(f"Graph file not found at {Paths.GRAPH_PATH.value}. Please build the graph first.") 
    return None  