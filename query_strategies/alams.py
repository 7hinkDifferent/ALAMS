import numpy as np
from .strategy import Strategy
from .least_confidence import LeastConfidence
from .kmeans_sampling import KMeansSampling
import torch
from .utils import kmeans_gaussian_scores, kmeans_gmm_scores

# Active Learning with Adaptive Mixture of Strategies (ALAMS)
# use gradient as post hoc attribution
# use relative improvement in validation loss as reward
# use exponential weights or MAB (UCB) to select strategy based on expected reward
# procedure:
# 1. query samples using weighted probabilities
# 2. update strategy with new samples
# 3. update: calculate gradient for each unlabeled sample based on previous model
# 4. train new model and record parameters update
# 5. calculate reward based on relative improvement in validation loss
# 6. calculate attribution score for each sample based on gradient and reward
# 7. update weights for each strategy based on attribution scores of samples queried by that strategy

class AdaptiveMixtureStrategiesGaussian(Strategy):
    def __init__(self, dataset, net):
        super(AdaptiveMixtureStrategiesGaussian, self).__init__(dataset, net)
        self.strategy_count = 2
        self.scores = np.array([1 / self.strategy_count for _ in range(self.strategy_count)])  # expected scores for each strategy
        self.gradients = None  # placeholder for gradients
        self.old_params = None  # placeholder for old parameters
        self.last_metrics = None
        self.update_rate = 0.9
        self.credit_mapping = [] # (strategy count x sample count)
        
    def update(self, pos_idxs, neg_idxs=None):
        # get new labeled samples and calculate gradients
        new_labeled_data = self.dataset.get_data_by_idxs(pos_idxs)
        self.gradients = self.net.compute_per_sample_gradients(new_labeled_data)
        self.old_params = self.net.get_flattened_params()
        return super().update(pos_idxs, neg_idxs)
        
    def feedback(self, info):
        meta = {}
        # define reward as relative improvement in training accuracy
        # update expected scores for each strategy based on attribution scores
        if self.old_params is not None:
            if self.last_metrics is not None:
                current_val_loss = info["best_val_loss"]
                reward = (self.last_metrics - current_val_loss) / self.last_metrics  # relative decrease in val loss
                # update expected scores for each strategy
                new_params = self.net.get_flattened_params()
                param_diff = new_params - self.old_params  # parameter update
                # calculate attribution scores for each sample
                attributions = []
                for grad in self.gradients:
                    attribution = torch.dot(grad, param_diff).item()
                    attributions.append(attribution)
                attributions = np.array(attributions)
                # normalize attributions with absolute sum
                if attributions.sum() != 0:
                    attributions = attributions / np.abs(attributions).sum()
                # TODO: whether or not normalize credits to [-1, 1]
                credits = reward * attributions

                # update expected scores with exponential weights
                delta = self.credit_mapping.dot(credits)  # (strategy count, )
                self.scores = np.array(self.scores) * np.exp(self.update_rate * delta)

                self.old_params = new_params  # update old params

                meta.update({
                    "attributions": attributions.tolist(),
                    "credits": credits.tolist(),
                    "reward": reward,
                    "score_update": delta.tolist(),
                    "updated_scores": self.scores.tolist()
                })
            
        self.last_metrics = info["best_val_loss"]
        return meta

    def query(self, n):
        info = {
            "selected_strategies": 0,
            "strategy_scores": 0.0,
        }
        # select samples based on expected scores

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Compute uncertainty scores
        probs, acc, loss = self.predict_prob(unlabeled_data)
        uncertainties = 1 - probs.max(1)[0]

        # Compute kmeans gmm scores
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        kmeans_gaussian_scores_values, cluster_info = kmeans_gaussian_scores(embeddings, n)

        # Combine scores
        combined_scores = self.scores[0] * uncertainties.numpy() + self.scores[1] * kmeans_gaussian_scores_values
        
        # q_idxs = combined_scores.argsort()[-n:][::-1] # argmax or probabilistic sampling
        # use probabilistic sampling based on expected scores
        q_idxs = np.random.choice(unlabeled_idxs, size=n, replace=False, p=combined_scores/combined_scores.sum())

        # Strategy attribution info (n_strategy x n_samples)
        self.credit_mapping = np.zeros((self.strategy_count, len(q_idxs)))
        for i in range(len(q_idxs)):
            self.credit_mapping[0][i] = uncertainties.numpy()[i]
            self.credit_mapping[1][i] = kmeans_gaussian_scores_values[i]
        # normalize credit mapping along strategies
        total = self.credit_mapping.sum(axis=0)
        self.credit_mapping = self.credit_mapping / (total[np.newaxis, :] + 1e-10)
        # for i in range(len(selected_idxs)):
        #     total = self.credit_mapping[:, i].sum()
        #     if total != 0:
        #         self.credit_mapping[:, i] = self.credit_mapping[:, i] / total

        return q_idxs, {
            "unlabeled_idxs": unlabeled_idxs.tolist(),
            "uncertainties": uncertainties.tolist(),
            "kmeans_gaussian_scores": kmeans_gaussian_scores_values.tolist(),
            "combined_scores": combined_scores.tolist(),
            "selected_unlabeled_idxs": q_idxs.tolist(),
            "score_weights": self.scores.tolist(),
            "credit_mapping": self.credit_mapping.tolist(),
            **cluster_info
        }


class AdaptiveMixtureStrategiesGMM(Strategy):
    def __init__(self, dataset, net):
        super(AdaptiveMixtureStrategiesGMM, self).__init__(dataset, net)
        self.strategy_count = 2
        self.scores = np.array([1 / self.strategy_count for _ in range(self.strategy_count)])  # expected scores for each strategy
        self.gradients = None  # placeholder for gradients
        self.old_params = None  # placeholder for old parameters
        self.last_metrics = None
        self.update_rate = 0.9
        self.credit_mapping = [] # (strategy count x sample count)
        
    def update(self, pos_idxs, neg_idxs=None):
        # get new labeled samples and calculate gradients
        new_labeled_data = self.dataset.get_data_by_idxs(pos_idxs)
        self.gradients = self.net.compute_per_sample_gradients(new_labeled_data)
        self.old_params = self.net.get_flattened_params()
        return super().update(pos_idxs, neg_idxs)
        
    def feedback(self, info):
        meta = {}
        # define reward as relative improvement in training accuracy
        # update expected scores for each strategy based on attribution scores
        if self.old_params is not None:
            if self.last_metrics is not None:
                current_val_loss = info["best_val_loss"]
                reward = (self.last_metrics - current_val_loss) / self.last_metrics  # relative decrease in val loss
                # update expected scores for each strategy
                new_params = self.net.get_flattened_params()
                param_diff = new_params - self.old_params  # parameter update
                # calculate attribution scores for each sample
                attributions = []
                for grad in self.gradients:
                    attribution = torch.dot(grad, param_diff).item()
                    attributions.append(attribution)
                attributions = np.array(attributions)
                # normalize attributions with absolute sum
                if attributions.sum() != 0:
                    attributions = attributions / np.abs(attributions).sum()
                # TODO: whether or not normalize credits to [-1, 1]
                credits = reward * attributions

                # update expected scores with exponential weights
                delta = self.credit_mapping.dot(credits)  # (strategy count, )
                self.scores = np.array(self.scores) * np.exp(self.update_rate * delta)
                # TODO: normalize scores
                # self.scores = self.scores / self.scores.sum()

                self.old_params = new_params  # update old params

                meta.update({
                    "attributions": attributions.tolist(),
                    "credits": credits.tolist(),
                    "reward": reward,
                    "score_update": delta.tolist(),
                    "updated_scores": self.scores.tolist()
                })
            
        self.last_metrics = info["best_val_loss"]
        return meta

    def query(self, n):
        info = {
            "selected_strategies": 0,
            "strategy_scores": 0.0,
        }
        # select samples based on expected scores

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # Compute uncertainty scores
        probs, acc, loss = self.predict_prob(unlabeled_data)
        uncertainties = 1 - probs.max(1)[0]

        # Compute kmeans gmm scores
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        kmeans_gmm_scores_values, cluster_info = kmeans_gmm_scores(embeddings, n)

        # Combine scores
        combined_scores = self.scores[0] * uncertainties.numpy() + self.scores[1] * kmeans_gmm_scores_values
        
        # q_idxs = combined_scores.argsort()[-n:][::-1] # argmax or probabilistic sampling
        # use probabilistic sampling based on expected scores
        q_idxs = np.random.choice(unlabeled_idxs, size=n, replace=False, p=combined_scores/combined_scores.sum())

        # Strategy attribution info (n_strategy x n_samples)
        self.credit_mapping = np.zeros((self.strategy_count, len(q_idxs)))
        for i in range(len(q_idxs)):
            self.credit_mapping[0][i] = uncertainties.numpy()[i]
            self.credit_mapping[1][i] = kmeans_gmm_scores_values[i]
        # normalize credit mapping along strategies
        total = self.credit_mapping.sum(axis=0)
        self.credit_mapping = self.credit_mapping / (total[np.newaxis, :] + 1e-10)
        # for i in range(len(selected_idxs)):
        #     total = self.credit_mapping[:, i].sum()
        #     if total != 0:
        #         self.credit_mapping[:, i] = self.credit_mapping[:, i] / total

        return q_idxs, {
            "unlabeled_idxs": unlabeled_idxs.tolist(),
            "uncertainties": uncertainties.tolist(),
            "kmeans_gmm_scores": kmeans_gmm_scores_values.tolist(),
            "combined_scores": combined_scores.tolist(),
            "selected_unlabeled_idxs": q_idxs.tolist(),
            "score_weights": self.scores.tolist(),
            "credit_mapping": self.credit_mapping.tolist(),
            **cluster_info
        }
