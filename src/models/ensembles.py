import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import Base_, BaseNet
from tqdm import trange
from ..data import MyDataloader
from icecream import ic

class GaussNLLNet(BaseNet):
    def __init__(self, p_dropout=0.0) -> None:
        super().__init__()
        nodes_per_layer = 16
        self.layers = nn.Sequential(
            nn.Linear(1, nodes_per_layer),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, 2)
        )

    def forward(self, x) -> torch.Tensor:
        res = self.layers(x)
        return res[..., [0]], F.softplus(res[..., [1]])
    
    def train_one_epoch(self, training_loader, optimizer, loss_fn=torch.nn.GaussianNLLLoss()):
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            loss = 0
            outputs, vars = self(inputs)
            #ic(outputs.shape)
            #ic(vars.shape)
            #ic(labels.shape)
            # Compute the loss and its gradients
            loss += loss_fn(outputs, labels, vars)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        return running_loss/i

class EnsembleNet(Base_):
    def __init__(self, base_net=GaussNLLNet, base_kwargs=dict(p_dropout=0.0), M=5) -> None:
        super().__init__()
        self.nets = nn.ModuleList([
            base_net(**base_kwargs) for _ in range(M)
        ])
        self.i = 0
        self.M = M
    
    def forward(self, x, mixture=False, *args, **kwargs):
        if mixture:
            y_means, y_vars = [], []
            for net in self.nets:
                mean, var = net(x,  *args, **kwargs)
                y_means.append(mean)
                y_vars.append(var)
            y_means, y_vars = torch.stack(y_means), torch.stack(y_vars) # batch nets,  batch nets
            y_mean = torch.mean(y_means, dim=0) # batch channel
            y_var = torch.mean(y_vars + y_means**2, dim=0) - y_mean ** 2
            return y_mean, y_var
        
        y_means = []
        for net in self.nets:
            mean = net(x)
            if len(mean) == 2:
                print("`net` output of length 2. Perhaps you meant to specific a mixture?")
            y_means.append(mean)
        y_means = torch.stack(y_means) # batch nets,  batch nets
        y_mean = torch.mean(y_means, dim=0) # batch channel
        y_var = torch.var(y_means, dim=0)
        return y_mean, y_var

    def train_model(self, epochs=1000, *args, **kwargs):
        self.train()
        for i, net in enumerate(self.nets):
            print(f"\tTraining net {i+1} of {len(self.nets)}.")
            net.train_model(epochs=epochs, *args, **kwargs)
        self.eval()
