import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
import os
from torch.utils.tensorboard import SummaryWriter

class Net:
    def __init__(self, net, params, device, logging_root="./logs"):
        self.net = net
        self.params = params
        self.device = device
        self.clf = self.net().to(device)
        self.logging_root = logging_root
        
    def reset(self):
        del self.clf
        self.clf = self.net().to(self.device)

    def train(self, train_data, val_data=None, round=0):
        logging_dir = f'{self.logging_root}/round_{round}'
        os.makedirs(logging_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f'{self.logging_root}/tensorboard')

        n_epoch = self.params['n_epoch']
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(train_data, shuffle=True, **self.params['train_args'])
        best_loss = 1e10
        best_epoch = 0
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            train_loss = 0.0
            train_acc = 0.0
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

                pred = out.max(1)[1]
                train_acc += (pred == y).sum().item()
                train_loss += loss.item() * x.shape[0]
            train_loss /= len(train_data)
            train_acc /= len(train_data)
            writer.add_scalar(f'round_{round}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'round_{round}/Train_Accuracy', train_acc, epoch)

            if val_data is not None:
                preds, val_loss = self.predict(val_data)
                val_acc = (preds == val_data.Y).sum().item() / len(val_data)
                writer.add_scalar(f'round_{round}/Val_Loss', val_loss, epoch)
                writer.add_scalar(f'round_{round}/Val_Accuracy', val_acc, epoch)
                # print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f
                #       }, Validation Accuracy: {val_acc:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    torch.save(self.clf.state_dict(), f'{logging_dir}/best_model.pth')

        torch.save(self.clf.state_dict(), f'{logging_dir}/last_model.pth')
        if val_data is not None:
            self.clf.load_state_dict(torch.load(f'{logging_dir}/best_model.pth', map_location=self.device))
            # print(f"Best model at epoch {best_epoch} with validation loss {best_loss:.4f}")

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        loss = 0.0
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
                loss += F.cross_entropy(out, y).item() * x.shape[0]
        loss /= len(data)
        return preds, loss
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        loss = 0.0
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                loss += F.cross_entropy(out, y).item() * x.shape[0]
        loss /= len(data)
        return probs, loss
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        loss = 0.0
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
                    loss += F.cross_entropy(out, y).item() * x.shape[0]
        loss /= (len(data) * n_drop)
        probs /= n_drop
        return probs, loss
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        loss = [0.0] * n_drop
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += prob.cpu()
                    loss[i] += F.cross_entropy(out, y).item() * x.shape[0]
        for i in range(n_drop):
            loss[i] /= len(data)
        return probs, loss
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
        

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

# need further implementation for deeper nets

# class VGG16_Net(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16_Net, self).__init__()
#         self.model = models.vgg16(pretrained=False)
#         self.model.classifier[6] = nn.Linear(4096, num_classes)

#     def forward(self, x):
#         x = self.model.features(x)
#         x = x.view(x.size(0), -1)
#         e1 = ...
#         return x, e1
    
# class ResNet18_Net(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18_Net, self).__init__()
#         self.model = models.resnet18(pretrained=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#         x = self.model.avgpool(x)
#         e1 = torch.flatten(x, 1)
#         x = self.model.fc(e1)
#         return x, e1