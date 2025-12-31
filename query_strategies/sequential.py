import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

# cluster first, then select one sample in each cluster based on uncertainty scores

class Sequential(Strategy):
    def __init__(self, dataset, net):
        super(Sequential, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Compute uncertainty scores
        probs, loss = self.predict_prob(unlabeled_data)
        uncertainties = 1 - probs.max(1)[0] # lower probs means more uncertain
        
        # KMeans clustering
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        cluster_idxs = cluster_learner.predict(embeddings)
        
        # Select the most uncertain sample in each cluster
        q_idxs = []
        for i in range(n):
            cluster_mask = cluster_idxs == i
            if cluster_mask.sum() > 0:
                cluster_samples = np.arange(embeddings.shape[0])[cluster_mask]
                cluster_uncertainties = uncertainties[cluster_mask]
                max_uncertainty_idx = cluster_uncertainties.argmax()
                q_idxs.append(cluster_samples[max_uncertainty_idx])
        
        q_idxs = np.array(q_idxs)
        
        return unlabeled_idxs[q_idxs], {
            "unlabeled_idxs": unlabeled_idxs.tolist(),
            "cluster_idxs": cluster_idxs.tolist(),
            "uncertainties": uncertainties.tolist(),
            "selected_unlabeled_idxs": unlabeled_idxs[q_idxs].tolist(),
            "selected_cluster_idxs": cluster_idxs[q_idxs].tolist(),
        }
