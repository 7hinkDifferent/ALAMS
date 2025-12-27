import numpy as np
from .strategy import Strategy
from .utils import rank_based_normalization, kmeans_gaussian_scores, kmeans_gmm_scores

# cluster first, then select one sample in each cluster based on uncertainty scores

class LinearCombination(Strategy):
    def __init__(self, dataset, net):
        super(LinearCombination, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        gaussian_scores, gaussian_meta = kmeans_gaussian_scores(embeddings, n)
        
        
