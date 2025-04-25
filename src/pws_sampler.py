import pandas as pd
import numpy as np

from .helper import set_all_seeds, compute_dataset_statistics, split_dtst, DF_KEYS

class PairWiseSampler:
    def __init__(self, last_fm_df:pd.DataFrame, batch_size:int, seed:int=42):
        """
        Initialize the PairWiseSampler with the LastFM DataFrame and batch size. 
        Splitting is done automatically by sorting the dataframe by timestamp.

        Args:
            last_fm_df (pd.DataFrame): LastFM DataFrame with the columns 'user_id', 'artist_id', 'track_id' and timestamp.
            batch_size (int): batch size for the sample
            seed (int, optional): set the random seed. Defaults to 42.
        """
        # Set the seed for reproducibility
        set_all_seeds(seed)

        # split the dataset into train, val and test sets
        train_df, val_df, test_df = split_dtst(last_fm_df, train_ratio=0.8)
        
 
        self.train = train_df

        # set variables for the string key in the dataframe for users and items
        U_KEY = DF_KEYS.USERS.value
        I_KEY = DF_KEYS.ITEMS.value

        # Re-map users and items so that they are represented by contiguous ids (users: 0...N-1, items: 0...M-1)
        self.user_map = {u: idx for idx, u in enumerate(self.train[U_KEY].unique())}
        self.item_map = {i: idx for idx, i in enumerate(self.train[I_KEY].unique())}
        self.train[U_KEY] = self.train[U_KEY].map(self.user_map)
        self.train[I_KEY] = self.train[I_KEY].map(self.item_map)

        # Calculate, for each user, the set of positive items
        self.user_pos = {u: self.train[self.train[U_KEY] == u][I_KEY].tolist() for u in self.train[U_KEY]}

        # Calculate, for each user, the number of positive items
        self.len_user_pos = {u: len(self.train[self.train[U_KEY] == u][I_KEY]) for u in self.train[U_KEY]}

        # Save some statistics for the sampling
        self.data_stats = compute_dataset_statistics(last_fm_df, verbose=False)

        # Save batch size
        self.batch_size = batch_size

        # Filter out interactions where user and/or item are not in the training set
        users = self.user_map.keys()
        items = self.item_map.keys()
        val_df = val_df[val_df[U_KEY].isin(list(users))]
        val_df = val_df[val_df[I_KEY].isin(list(items))]
        test_df = test_df[test_df[U_KEY].isin(list(users))]
        test_df = test_df[test_df[I_KEY].isin(list(items))]

        # Map users and items in val_df and test
        val_df[U_KEY] = val_df[U_KEY].map(self.user_map)
        val_df[I_KEY] = val_df[I_KEY].map(self.item_map)
        test_df[U_KEY] = test_df[U_KEY].map(self.user_map)
        test_df[I_KEY] = test_df[I_KEY].map(self.item_map)

        # Get relevant items
        self.relevant_items_val = dict((u, set(val_df[val_df[U_KEY] == u][I_KEY].tolist())) for u in val_df[U_KEY].unique())
        self.relevant_items_test = dict((u, set(test_df[test_df[U_KEY] == u][I_KEY].tolist())) for u in test_df[U_KEY].unique())
        self.num_users_val = len(self.relevant_items_val)
        self.num_users_test = len(self.relevant_items_test)

    # Create an iterator that returns batches of (u, i, j), where u --> user, i --> positive item, j --> negative items over user set
    def step(self):
        r_int = np.random.randint

        def sample():
            #u = r_int(self.data_stats['num_users'])
            u = r_int(len(self.train['user_id'].unique()))
            ui = self.user_pos[u]
            lui = self.len_user_pos[u]
            if lui == self.data_stats['num_items']:
                sample()
            i = ui[r_int(lui)]

            j = r_int(self.data_stats['num_items'])
            while j in ui:
                j = r_int(self.data_stats['num_items'])
            return u, i, j

        for batch_start in range(0, len(self.train), self.batch_size):
            bui, bii, bij = zip(*[sample() for _ in range(batch_start, min(batch_start + self.batch_size, self.data_stats['num_interactions']))])
            yield bui, bii, bij