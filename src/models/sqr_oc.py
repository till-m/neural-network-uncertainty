import torch
import torch.nn as nn
from tqdm import trange
from ..data import MyDataloader
from .base import BaseNet


class PinballLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, quantiles):
        error = target - pred
        upper =  quantiles * error
        lower = (quantiles - 1) * error 

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss


class SQRNet(BaseNet):
    def __init__(self, p_dropout=0.5) -> None:
        super().__init__()
        nodes_per_layer = 16
        self.layers = nn.Sequential(
            nn.Linear(2, nodes_per_layer),
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

    def forward(self, x, tau=None) -> torch.Tensor:
        if tau is not None:
            if type(tau) != torch.Tensor or tau.shape != x.shape:
                tau = torch.ones_like(x) * tau
            return self.layers(torch.cat((x, tau), axis=-1))
        tau = torch.ones_like(x) * 0.5
        return self.layers(torch.cat((x, tau), axis=-1))

    
    def train_one_epoch(self, training_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn=PinballLoss()):
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            tau = torch.rand_like(labels)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self(inputs, tau)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels, tau)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        return running_loss/i


class OCNet(BaseNet):
    def __init__(self, p_dropout=0.5, k=10, _lambda=1.) -> None:
        super().__init__()
        nodes_per_layer = 16
        self.k = k
        self._lambda = _lambda
        self.layers = nn.Sequential(
            nn.Linear(1, nodes_per_layer),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.Dropout(p=p_dropout),
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(nodes_per_layer, 1)
        )
        #self.certificate = nn.utils.parametrizations.orthogonal(nn.Linear(nodes_per_layer, self.k))
        self.certificate = nn.Linear(nodes_per_layer, self.k, bias=True)
        # expected response of certificates over an in-domain sample (not part of training set)
        self.expected_response = None

    def forward(self, x, return_certificate_response=False) -> torch.Tensor:  
        x = self.layers(x)
        if return_certificate_response:
            return self.output_layer(x), self.certificate(x)
        return self.output_layer(x)

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
            loss_model = loss_fn(outputs, labels)
            loss_model.backward(retain_graph=True)
            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss_model.item()

        return running_loss/i
    
    def train_certificates(self, training_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn=torch.nn.MSELoss()):
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        n_batches = len(training_loader)
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, _ = data

            if n_batches-1 == i:
                # Final batch -- use for calibration
                _, certificate_response = self(inputs, return_certificate_response=True)
                self.expected_response = torch.mean(certificate_response**2)
                print("Finished training certificates")
                return running_loss/i

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            _, certificate_response = self(inputs, return_certificate_response=True)

            loss_certs = (
                loss_fn(certificate_response, torch.zeros_like(certificate_response))
                + self._lambda * torch.mean(torch.abs(torch.eye(self.k) - self.certificate.weight @ self.certificate.weight.T))
            )
            loss_certs.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss_certs.item()
    
    
    def train_model(self, epochs=1000):
        self.train()
        training_loader = MyDataloader(batch_size=128, shuffle=True)
        optimizer = torch.optim.AdamW([*self.output_layer.parameters(), *self.layers.parameters()], weight_decay=1e-2)
        optimizer_cert = torch.optim.Adam(self.certificate.parameters())

        pbar = trange(epochs)
        for i in pbar:
            epoch_loss = self.train_one_epoch(training_loader, optimizer)
            pbar.set_description(f"Epoch {i}, prev. loss {epoch_loss:.3f}")
        cert_loss = self.train_certificates(training_loader, optimizer_cert)
        print(f"Certificate loss {cert_loss:.3f}")
        self.eval()

