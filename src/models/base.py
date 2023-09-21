import torch
import torch.nn as nn
from typing import Tuple
from tqdm import trange
from ..data import MyDataloader


class Base_(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self) -> torch.Tensor:
        raise NotImplementedError
    
    def train_one_epoch(self, training_loader, optimizer, loss_fn):
        raise NotImplementedError
    
    def train_model(self, epochs=1000):
        self.train()
        training_loader = MyDataloader(batch_size=128, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-2)

        pbar = trange(epochs)
        for i in pbar:
            epoch_loss = self.train_one_epoch(training_loader, optimizer)
            pbar.set_description(f"Epoch {i}, prev. loss {epoch_loss:.3f}")
        self.eval()

class BaseNet(Base_):
    def __init__(self, p_dropout=0.1) -> None:
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
            nn.Linear(nodes_per_layer, 1)
        )

    def forward(self, x) -> torch.Tensor:
        return self.layers(x)
    
    def forward_var(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)
    
    def train_one_epoch(self, training_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn=torch.nn.MSELoss()):
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
            outputs = self(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        return running_loss/i