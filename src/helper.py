import random
import numpy as np
import torch
import pandas as pd
from enum import Enum

class DF_KEYS(Enum):
    USERS = "user_id"
    ITEMS = "artist_id"

def set_all_seeds(random_seed):
    # Set seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def evaluate(preds, num_users_val, num_users_test, relevant_items_val, relevant_items_test):

    metrics = dict()

    # Compute Recall@k and Precision@k
    recall_val = 0.0
    precision_val = 0.0
    ndcg_val = 0.0

    recall_test = 0.0
    precision_test = 0.0
    ndcg_test = 0.0

    recommended = preds.shape[-1]

    # Validation
    for user, rel_items in relevant_items_val.items():
        recommended_items = set(preds[user, :].tolist())
        relevant_val = len(rel_items)
        relevant_in_recommended_val = len(rel_items & recommended_items)
        # Recall@k
        recall_val += relevant_in_recommended_val / relevant_val
        # Precision@k
        precision_val += relevant_in_recommended_val / recommended
        # nDCG@k
        idcg = sum((1 / np.log2(idx + 2)) for idx, _ in enumerate(list(rel_items)[:recommended]))
        dcg = sum((1 / np.log2(idx + 2)) for idx, item in enumerate(recommended_items) if item in rel_items)
        ndcg_val += (dcg / idcg) if dcg > 0 else 0.0

    # Test
    for user, rel_items in relevant_items_test.items():
        recommended_items = set(preds[user, :].tolist())
        relevant_test = len(rel_items)
        relevant_in_recommended_test = len(rel_items & recommended_items)
        # Recall@k
        recall_test += relevant_in_recommended_test / relevant_test
        # Precision@k
        precision_test += relevant_in_recommended_test / recommended
        # nDCG@k
        idcg = sum((1 / np.log2(idx + 2)) for idx, _ in enumerate(list(rel_items)[:recommended]))
        dcg = sum((1 / np.log2(idx + 2)) for idx, item in enumerate(recommended_items) if item in rel_items)
        ndcg_test += (dcg / idcg) if dcg > 0 else 0.0

    recall_val /= num_users_val
    recall_test /= num_users_test
    precision_val /= num_users_val
    precision_test /= num_users_test
    ndcg_val /= num_users_val
    ndcg_test /= num_users_test

    metrics['recall_val'] = recall_val
    metrics['recall_test'] = recall_test
    metrics['precision_val'] = precision_val
    metrics['precision_test'] = precision_test
    metrics['ndcg_val'] = ndcg_val
    metrics['ndcg_test'] = ndcg_test

    return metrics

def compute_dataset_statistics(df:pd.DataFrame, verbose=True):
    # Read the dataset with pandas

    # Calculate number of users, items, interactions, density, and sparsity.
    dataset_statistics = {}
    num_users = df[DF_KEYS.USERS.value].nunique()
    dataset_statistics['num_users'] = num_users
    num_items = df[DF_KEYS.ITEMS.value].nunique()
    dataset_statistics['num_items'] = num_items
    num_interactions = len(df)
    dataset_statistics['num_interactions'] = num_interactions
    density = num_interactions / (num_items * num_users)
    dataset_statistics['density'] = density
    sparsity = 1 - density
    dataset_statistics['sparsity'] = sparsity

    # Print them
    if verbose:
        print(f'Number of users: {num_users}')
        print(f'Number of items: {num_items}')
        print(f'Number of interactions: {num_interactions}')
        print(f'Density: {density}')
        print(f'Sparsity: {sparsity}')

    return dataset_statistics

def split_dtst(df:pd.DataFrame, train_ratio:float=0.8)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets.
    The split is done by timestamp, such that the train set contains the first interactions and the test set contains the last interactions.

    Args:
        df (pd.DataFrame): dataset to split with a timestamp column
        train_ratio (float, optional): how much goes in the train dataset compared to the 2 others. Defaults to 0.8.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
    """
    # Sort the dataframe by timestamp
    sorted_df = df.sort_values(ascending=True,by='timestamp')

    # Calculate the split index
    train_split_idx = int(len(df) * train_ratio)
    val_split_idx = int(len(df) * (train_ratio + (1 - train_ratio) / 2))

    # Split the dataframe into train and test sets
    train_df = sorted_df.iloc[:train_split_idx]
    val_df = sorted_df.iloc[train_split_idx:val_split_idx]
    test_df = sorted_df.iloc[val_split_idx:]

    return train_df, val_df, test_df
