
import torch
from tqdm import trange
from ..data import MyDataloader
from .base import Base_, BaseNet
from torch.optim.swa_utils import SWALR
from .swag_utils import SWAG

class SWAGModel(Base_):
    def __init__(self, model=BaseNet, model_kwargs=dict(p_dropout=0.0), max_num_models=20):
        super().__init__()
        self.model = model(**model_kwargs)

        self.cov_mat = True
        self.max_num_models = max_num_models
        self.swag_network = SWAG(model, no_cov_mat=not self.cov_mat, max_num_models=self.max_num_models,  **model_kwargs)
        
    def forward(self, x, return_var=False, n_samples=30):
        if return_var:
            y_means, y_vars = [], []
            for i in range(n_samples):
                self.swag_network.sample(scale=0.5, cov=self.cov_mat)
                mean = self.swag_network(x)
                y_means.append(mean)
            y_means = torch.stack(y_means) # batch nets,  batch nets
            return torch.mean(y_means, dim=0), torch.var(y_means, dim=0)
        return self.model(x)

    def train_model(self, epochs=1000, swa_start=0.75, swa_lr=5e-2, lr_init=1e-3, moment_update_freq=7):
        swa_start = int(epochs * swa_start)
    
        self.train()
        training_loader = MyDataloader(batch_size=128, shuffle=True)
        #optimizer = torch.optim.SGD(self.parameters(), weight_decay=1e-2, lr=lr_init)
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-2, lr=lr_init)
        scheduler = SWALR(optimizer, swa_lr=swa_lr)

        pbar = trange(epochs)
        for i in pbar:
            epoch_loss = self.model.train_one_epoch(training_loader, optimizer)
            pbar.set_description(f"Epoch {i}, prev. loss {epoch_loss:.3f}")

            if (i + 1) > swa_start and (i % moment_update_freq) == 0:
                self.swag_network.collect_model(self.model)
                scheduler.step()
        self.eval()
