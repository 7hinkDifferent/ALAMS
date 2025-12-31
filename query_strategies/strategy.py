import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def feedback(self, info):
        pass

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def reset_net(self):
        self.net.reset()

    def train(self, round=0):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        val_data = self.dataset.get_val_data()
        train_results = self.net.train(labeled_data, val_data, round=round)
        return train_results

    def predict(self, data):
        preds, loss = self.net.predict(data)
        return preds, loss

    def predict_prob(self, data):
        probs, loss = self.net.predict_prob(data)
        return probs, loss

    def predict_prob_dropout(self, data, n_drop=10):
        probs, loss = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs, loss

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs, loss = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs, loss
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

