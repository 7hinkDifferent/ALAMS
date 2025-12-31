import numpy as np
from .strategy import Strategy
from .utils import kmeans_gaussian_scores, kmeans_gmm_scores

# weighted sum of uncertainty scores and kmeans gaussian / gmm scores
# s = alpha * uncertainty_score + (1 - alpha) * kmeans_gaussian_score

class WeightedSumGaussian(Strategy):
    def __init__(self, dataset, net):
        super(WeightedSumGaussian, self).__init__(dataset, net)

    def query(self, n):
        alpha = 0.5  # weight for uncertainty scores

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Compute uncertainty scores
        probs, loss = self.predict_prob(unlabeled_data)
        uncertainties = 1 - probs.max(1)[0]

        # Compute kmeans gaussian scores
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        kmeans_gaussian_scores_values, cluster_info = kmeans_gaussian_scores(embeddings, n)

        # Combine scores
        combined_scores = alpha * uncertainties.numpy() + (1 - alpha) * kmeans_gaussian_scores_values
        q_idxs = combined_scores.argsort()[-n:][::-1]
        return unlabeled_idxs[q_idxs], {
            "unlabeled_idxs": unlabeled_idxs.tolist(),
            "uncertainties": uncertainties.tolist(),
            "kmeans_gaussian_scores": kmeans_gaussian_scores_values.tolist(),
            "combined_scores": combined_scores.tolist(),
            "selected_unlabeled_idxs": unlabeled_idxs[q_idxs].tolist(),
            **cluster_info
        }
    
class WeightedSumGMM(Strategy):
    def __init__(self, dataset, net):
        super(WeightedSumGMM, self).__init__(dataset, net)

    def query(self, n):
        alpha = 0.5  # weight for uncertainty scores

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Compute uncertainty scores
        probs, loss = self.predict_prob(unlabeled_data)
        uncertainties = 1 - probs.max(1)[0]

        # Compute kmeans gmm scores
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        kmeans_gmm_scores_values, cluster_info = kmeans_gmm_scores(embeddings, n)

        # Combine scores
        combined_scores = alpha * uncertainties.numpy() + (1 - alpha) * kmeans_gmm_scores_values
        q_idxs = combined_scores.argsort()[-n:][::-1]
        return unlabeled_idxs[q_idxs], {
            "unlabeled_idxs": unlabeled_idxs.tolist(),
            "uncertainties": uncertainties.tolist(),
            "kmeans_gmm_scores": kmeans_gmm_scores_values.tolist(),
            "combined_scores": combined_scores.tolist(),
            "selected_unlabeled_idxs": unlabeled_idxs[q_idxs].tolist(),
            **cluster_info
        }