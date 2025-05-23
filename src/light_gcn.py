
import torch
import torch_geometric
from torch_geometric.nn import LGConv
import numpy as np
import pandas as pd
from torch_sparse import SparseTensor
import math
from tqdm import tqdm

from .helper import set_all_seeds, evaluate, DF_KEYS
from .pws_sampler import PairWiseSampler


class LightGCNModel(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_size, num_layers, adj, normalize, seed=42, **kwargs):
        super().__init__()

        # Set all seeds
        set_all_seeds(seed)

        self.num_users = num_users
        self.num_items = num_items

        # Initialize the user and item embeddings
        self.Gu = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_size)
        self.Gi = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_size)
        torch.nn.init.normal_(self.Gu.weight, std=0.1)
        torch.nn.init.normal_(self.Gi.weight, std=0.1)

        # LightGCN layers
        self.n_layers = num_layers
        self.adj = adj
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=normalize), 'x, edge_index -> x'))
        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)

    # Message-passing function
    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.Gu.weight, self.Gi.weight), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer], self.adj)]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer], self.adj)]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = torch.mean(torch.stack(all_embeddings, 0), dim=0)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    # Forward function
    @staticmethod
    def forward(inputs):
        gu, gi = inputs
        xui = torch.sum(gu * gi, 1)
        return xui

    # Train step for each batch
    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[torch.tensor(user)], gi[torch.tensor(pos)]))
        xu_neg = self.forward(inputs=(gu[torch.tensor(user)], gi[torch.tensor(neg)]))
        return xu_pos, xu_neg

    # Predict function for the evaluation
    def predict(self, gu, gi, **kwargs):
        return torch.sigmoid(torch.matmul(gu, torch.transpose(gi, 0, 1)))


class LightGCN:
    def __init__(self,
                 df:pd.DataFrame,
                 batch_size:int,
                 embedding_size:int,
                 num_layers:int,
                 normalize:bool,
                 learning_rate:float,
                 regularization:float,
                 top_k:int=20,
                 seed:int=42):
        # Initialize the sampler
        self.pws = PairWiseSampler(df, batch_size=batch_size, seed=seed)

        # initialize user and item keys
        U_KEY = DF_KEYS.USERS.value
        I_KEY = DF_KEYS.ITEMS.value

        # Training mask (for evaluation)
        self.train_mask = torch.zeros(self.pws.data_stats['num_users'], self.pws.data_stats['num_items'], dtype=torch.bool)
        pos_users = self.pws.train[U_KEY].tolist()
        pos_items = self.pws.train[I_KEY].tolist()
        self.train_mask[pos_users, pos_items] = True

        # Create the adjacency matrix for the bipartite and undirected user-item graph
        rows = pos_users
        cols = [it + self.pws.data_stats['num_users'] for it in pos_items]
        edge_index = np.array([rows, cols])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self.pws.data_stats['num_users'] + self.pws.data_stats['num_items'],
                                              self.pws.data_stats['num_users'] + self.pws.data_stats['num_items']))

        # Initialize the model
        self.model = LightGCNModel(embedding_size=embedding_size,
                                   num_users=self.pws.data_stats['num_users'],
                                   num_items=self.pws.data_stats['num_items'],
                                   num_layers=num_layers,
                                   normalize=normalize,
                                   adj=self.adj,
                                   seed=seed)

        # Instantiate the optimizer (e.g., Adam)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Save the regularization coefficient for later
        self.reg = regularization

        # Save top_k for evaluation
        self.top_k = top_k

    def fit(self, epochs, val_epoch):
        # Put model into train mode
        self.model.train()

        best_recall_val = -np.inf
        best_recall_test = 0.0
        best_precision_test = 0.0
        best_ndcg_test = 0.0
        best_epoch = 1

        # Train over all the epochs
        for epoch in range(epochs):
            steps = 0
            print('**************************************************************')
            print(f'Epoch {epoch + 1}/{epochs}')

            transactions = self.pws.data_stats['num_interactions']
            batch_size = self.pws.batch_size
            n_batch = int(transactions / batch_size) if transactions % batch_size == 0 else int(transactions / batch_size) + 1

            with tqdm(total=n_batch) as t:
                for batch in self.pws.step():
                    steps += 1

                    xui, xuj = self.model.train_step(batch)
                    user, pos, neg = batch
                    loss = torch.mean(torch.nn.functional.softplus(xuj - xui))
                    reg_loss = self.reg * (1 / 2) * (self.model.Gu.weight[torch.tensor(user)].norm(2).pow(2) +
                                                     self.model.Gi.weight[torch.tensor(pos)].norm(2).pow(2) +
                                                     self.model.Gi.weight[torch.tensor(neg)].norm(2).pow(2)) / len(user)

                    loss += reg_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'BPR loss': f'{loss / steps:.5f}'})
                    t.update()

            if epoch % val_epoch == 0:
                preds = torch.empty(self.pws.data_stats['num_users'], self.top_k, dtype=torch.long)
                gu, gi = self.model.propagate_embeddings(evaluate=True)
                for index, offset in enumerate(range(0, self.pws.data_stats['num_users'], self.pws.batch_size)):
                    offset_stop = min(offset + self.pws.batch_size, self.pws.data_stats['num_users'])
                    predictions = self.model.predict(gu[offset: offset_stop], gi)
                    predictions[self.train_mask[offset:offset_stop]] = -float("inf")
                    _, sorted_pred = torch.topk(predictions, self.top_k)
                    preds[offset:offset_stop] = sorted_pred
                metrics = evaluate(preds=preds.cpu().numpy(),
                                   num_users_val=self.pws.num_users_val,
                                   num_users_test=self.pws.num_users_test,
                                   relevant_items_val=self.pws.relevant_items_val,
                                   relevant_items_test=self.pws.relevant_items_test)
                print(f'Validation: Recall@{self.top_k}={metrics["recall_val"]:.4f}\tPrecision@{self.top_k}={metrics["precision_val"]:.4f}\tnDCG@{self.top_k}={metrics["ndcg_val"]:.4f}')
                print(f'Test: Recall@{self.top_k}={metrics["recall_test"]:.4f}\tPrecision@{self.top_k}={metrics["precision_test"]:.4f}\tnDCG@{self.top_k}={metrics["ndcg_test"]:.4f}')

                if metrics["recall_val"] > best_recall_val:
                    best_recall_val = metrics["recall_val"]
                    best_recall_test = metrics["recall_test"]
                    best_precision_test = metrics["precision_test"]
                    best_ndcg_test = metrics["ndcg_test"]
                    best_epoch = epoch + 1

        print('\nTRAINING COMPLETE!')
        print(f'Best epoch: {best_epoch}')
        print(f'Test metrics: Recall@{self.top_k}={best_recall_test:.4f}\tPrecision@{self.top_k}={best_precision_test:.4f}\tnDCG@{self.top_k}={best_ndcg_test:.4f}')

        return preds.cpu().numpy()