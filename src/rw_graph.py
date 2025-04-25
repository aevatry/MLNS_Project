import pandas as pd
import networkx as nx
import numpy as np
import math

from .helper import set_all_seeds
from .pws_sampler import PaiselfiseSampler


class Random_Walk:
    def __init__(self, G:nx.Graph, df:pd.DataFrame, beta:float, steps:int=3, train_ratio:float=1.2, top_k:int=20, seed:int=42, max_iterations:int = 200_000):
        """
        Initialize the Random Walk model.
        This model is used to make predictions of recommandable items based on a knowledge graph.

        Args:
            G (nx.Graph): Full graph of the dataset
            df (pd.DataFrame): Paiselfise iteractions of users and items of the dataset
            beta (float): Beta parameter for the RP3-beta model
            steps (int, Optional): Power parameter for the RP3-beta model. Defaults to 3.
            train_ratio (float, Optional): ratio of train to test/val interactions between users and items. Defaults to 0.8.
            top_k (int, Optional): number of top results to include in evaluation. Defaults to 20.
            seed (int, Optional): Random seed. Defaults to 42.
            max_iterations (int, Optional): The max number of individual random walks to perform during the search. Defaults to 200_000
        """
        # Set all seeds
        set_all_seeds(seed)

        # Initialize the sampler
        self.pws = PaiselfiseSampler(df, batch_size=512, seed=seed)

        self.G = G.copy()
        self.beta = beta
        self.steps = steps
        self.train_ratio = train_ratio
        self.top_k = top_k 
        self.max_iters = max_iterations


        print(f"Initially, graph had: {self.G.number_of_edges()} edges and {self.G.number_of_nodes()} nodes")
        self.mask_graph()
        print(f"After masking, graph has: {self.G.number_of_edges()} edges and {self.G.number_of_nodes()} nodes")

    def mask_graph(self)->None:

        # remove the nodes (user or item) that are not used to make predictions 
        nodes_to_remove = []
        for node in self.G.nodes():
            if (node not in self.pws.user_map.keys() and
                node not in self.pws.item_map.keys()):
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.G.remove_node(node)

        # remove the edges we want to make the prediction over. here remove items in test set
        for u in self.pws.relevant_items_test.keys():
            # list of items mapped to integers that correspond to the user u in validation dataset
            list_i = self.pws.relevant_items_test[u]

            # graph key of node u (u is an integer that mapped in the paiselfise sampler) 
            key_node_u = list(self.pws.user_map.keys())[list(self.pws.user_map.values()).index(u)]

            for item in list_i:
                # graph key of node i (i is an integer that mapped in the paiselfise sampler) 
                key_node_i = list(self.pws.item_map.keys())[list(self.pws.item_map.values()).index(item)]
                self.G.remove_edge(key_node_u, key_node_i)
        
        # remove the edges we want to make the prediction over. here remove items in validation set
        for u in self.pws.relevant_items_val.keys():
            # list of items mapped to integers that correspond to the user u in validation dataset
            list_i = self.pws.relevant_items_val[u]

            # graph key of node u (u is an integer that mapped in the paiselfise sampler) 
            key_node_u = list(self.pws.user_map.keys())[list(self.pws.user_map.values()).index(u)]

            for item in list_i:
                # graph key of node i (i is an integer that mapped in the paiselfise sampler) 
                key_node_i = list(self.pws.item_map.keys())[list(self.pws.item_map.values()).index(item)]
                self.G.remove_edge(key_node_u, key_node_i)

    def predict(self, user_node:str, eps:float=0.005)->np.ndarray:
        """
        Predict the top K items for a given user node using the random walk algorithm described in [Updatable, accurate, diverse, and scalable recommendations for interactive applications](https://dl.acm.org/doi/10.1145/2955101), based on the transition probability matrix of the graph.

        This method is approximated using the RP3-beta algorithm. An exact method exists (taking the Markov transition matrix and computing the s-power of that matrix) but is computationally too expensive.

        This random walk approach also allows us to accomodate for the fact that our graph is not exactly bipartite (i.e. there are edges between items and items).

        Args:
            user_node (str): start node
            eps (float, optional): Convergeance threshold. Convergeance is calculated as the mean of: |I_n+1 - I_n|/I_n or the percentage change in the score of the obtained by random walk Defaults to 0.005.

        Returns:
            np.ndarray: top k predicted items for the user (where k is defined in the class initialization)
        """
        iteration = 0
        convergeance_not_reached = True

        # initialize vectors that will hold scores at different steps
        rp_iter_n = np.zeros(self.G.number_of_nodes(), dtype=np.float32)
        rp_iter_n_plus_1 = np.zeros(shape=self.G.number_of_nodes(), dtype=np.float32)

        while convergeance_not_reached and iteration<self.max_iters:
            # perform a random walk
            node_idx, score = self.perform_1_walk(self.G, self.beta, self.steps, user_node)

            # update the scores
            rp_iter_n_plus_1[node_idx] = rp_iter_n_plus_1[node_idx] + score

            # check for convergence
            has_reached = self.check_convergeance(rp_iter_n_plus_1, rp_iter_n, node_idx, score, eps)
            if has_reached:
                convergeance_not_reached = False
                print(f"Convergeance reached after {iteration} iterations")
                
            # update the previous iteration scores
            rp_iter_n[node_idx] = rp_iter_n[node_idx] + score

            iteration += 1

        top_k_items = self.get_top_k_items(rp_iter_n)
        return top_k_items
    
    def get_top_k_items(self, rp:np.ndarray)->np.ndarray:
        
        ranked_indices = np.argsort(rp)[::-1]
        top_k_items = []

        curr_idx = 0
        while len(top_k_items) < self.top_k and curr_idx < len(ranked_indices):
            node_idx = ranked_indices[curr_idx]
            node = list(self.G.nodes())[node_idx]
            node_type = self.G.nodes[node]['node_type']
            if node_type == 'user':
                # skip user nodes
                curr_idx += 1
                continue
            node_item_maped = self.pws.item_map[node]
            if node_item_maped in self.pws.user_pos[node_item_maped]:
                # skip items that are already in the user positive items
                curr_idx += 1
                continue

            # item not a user or already in the direct neighborhood of the user in training set
            top_k_items.append(node)
            curr_idx += 1
        
        return np.array(top_k_items)

    def sample_user_node(self)->str:
        """
        Sample a random user from the graph

        Returns:
            str: _description_
        """
        possible_nodes = []
        for node in self.G.nodes():
            if (self.G.nodes[node]['node_type'] == 'user' and
                self.pws.user_map[node] in self.pws.relevant_items_val.keys() and
                self.pws.user_map[node] in self.pws.relevant_items_test.keys()):
                possible_nodes.append(node)
        user_node = np.random.choice(possible_nodes)
        return user_node
    
    @staticmethod
    def perform_1_walk(G:nx.Graph, beta:float, steps:int, user_node:str)->tuple[int, float]:
        """
        Perform a single random walk on the graph G starting from the user node.

        Args:
            G (nx.Graph): Graph to perform the random walk on.
            beta (float): Beta parameter for the RP3-beta model.
            steps (int): Number of steps to take in the random walk.
            user_node (str): Starting node for the random walk.
            num_items (int): number of items in the graph

        Returns:
            tuple[int, float]: index of node random walk finished one, score to be added to the node
        """

        # Perform the random walk
        current_node = user_node
        for _ in range(steps):
            # Get the neighbors of the current node
            neighbors = list(G.neighbors(current_node))
            # Choose a random neighbor
            current_node = np.random.choice(neighbors)

        # make sure the node is not a user
        while G.nodes[current_node]['node_type'] == 'user':
            neighbors = list(G.neighbors(current_node))
            current_node = np.random.choice(neighbors)
        
        # find index of the node in the graph
        node_idx = list(G.nodes()).index(current_node)
        # find the score of the node in the graph
        score = 1 / math.pow(G.degree(current_node), beta)

        return node_idx, score
    
    @staticmethod
    def check_convergeance(rp_iter_n_plus_1:np.ndarray, rp_iter_n:np.ndarray, score_idx:int, score:float, eps:float)->bool:
        """
        Check convergeance of the random walk algorithm.

        Args:
            rp_iter_n_plus_1 (np.ndarray): scores at the current iteration
            rp_iter_n (np.ndarray): scores at the previous iteration
            score_idx (int): idx of the node score that was updated
            score (float): amount of the score update
            eps (float): convergeance threshold

        Returns:
            bool: Wether the convergeance was reached or not
        """
        curr_score = rp_iter_n_plus_1[score_idx]
        prev_score = rp_iter_n[score_idx]

        if prev_score == 0:
            return False
        
        update = abs(curr_score - prev_score) / prev_score
        if update < eps:
            return True

        return False
    
    def evaluate_1_user(self, user_node:str, preds:np.ndarray)->dict:

        recommanded_in_pws = []
        for item in preds:
            recommanded_in_pws.append(self.pws.item_map[item])
        recommanded_in_pws = np.array(recommanded_in_pws)

        metrics = dict()

        recommended = preds.shape[-1]

        recommended_items = set(preds.tolist())
        user_idx_pws = self.pws.user_map[user_node]
        
        # Validation
        rel_items = self.pws.relevant_items_val[user_idx_pws]
        
        relevant_val = len(rel_items)
        relevant_in_recommended_val = len(rel_items & recommended_items)
        # Recall@k
        recall_val = relevant_in_recommended_val / relevant_val
        # Precision@k
        precision_val = relevant_in_recommended_val / recommended

        # Test
        rel_items = self.pws.relevant_items_test[user_idx_pws]

        relevant_test = len(rel_items)
        relevant_in_recommended_test = len(rel_items & recommended_items)
        # Recall@k
        recall_test = relevant_in_recommended_test / relevant_test
        # Precision@k
        precision_test= relevant_in_recommended_test / recommended


        metrics['recall_val'] = recall_val
        metrics['recall_test'] = recall_test
        metrics['precision_val'] = precision_val
        metrics['precision_test'] = precision_test

        return metrics

        
        
        

        

        