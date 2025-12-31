import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num, full_init=False):
        tmp_idxs = np.arange(self.n_pool)
        if full_init:
            self.labeled_idxs[:] = True
            return tmp_idxs
        # generate initial labeled pool
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        return tmp_idxs[:num]
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_data_by_idxs(self, idxs):
        return self.handler(self.X_train[idxs], self.Y_train[idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_val_data(self):
        return self.handler(self.X_val, self.Y_val)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def get_statistics(self):
        return {
            'train': {
                'total': len(self.Y_train),
                'class_count': torch.bincount(torch.tensor(self.Y_train).detach().clone()).tolist()
            },
            'val': {
                'total': len(self.Y_val),
                'class_count': torch.bincount(torch.tensor(self.Y_val).detach().clone()).tolist()
            },
            'test': {
                'total': len(self.Y_test),
                'class_count': torch.bincount(torch.tensor(self.Y_test).detach().clone()).tolist()
            },
            'labeled': {
                'total': self.labeled_idxs.sum().item(),
                'class_count': torch.bincount(torch.tensor(self.Y_train[self.labeled_idxs]).detach().clone()).tolist()
            },
            'unlabeled': {
                'total': (~self.labeled_idxs).sum().item(),
                'class_count': torch.bincount(torch.tensor(self.Y_train[~self.labeled_idxs]).detach().clone()).tolist()
            }
        }

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:50000], raw_train.targets[:50000], raw_train.data[50000:], raw_train.targets[50000:], raw_test.data, raw_test.targets, handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:50000], raw_train.targets[:50000], raw_train.data[50000:], raw_train.targets[50000:], raw_test.data, raw_test.targets, handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:50000], torch.from_numpy(data_train.labels)[:50000], data_train.data[50000:], torch.from_numpy(data_train.labels[50000:]), data_test.data, torch.from_numpy(data_test.labels), handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_train.data[40000:], torch.LongTensor(data_train.targets[40000:]), data_test.data, torch.LongTensor(data_test.targets), handler)
