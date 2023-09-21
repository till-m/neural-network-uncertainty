import torch
import torch.nn as nn
from typing import Tuple
import einops

from .base import BaseNet

class MCDropoutNet(BaseNet):
    def __init__(self, p_dropout=0.5, weight_decay=1e-2, l=50., N=2_000) -> None:
        super().__init__(p_dropout=p_dropout)
        self.weight_decay = weight_decay
        self.l = l
        self.N = N
        self.p_dropout = p_dropout
    
    def forward(self, x, return_var=False, n=20) -> Tuple[torch.Tensor, torch.Tensor]:
        if return_var:
            for m in self.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
            res = []
            for _ in range(n):
                res.append(self(x))
            res = torch.stack(res)
            y_mean = torch.mean(res, axis=0)
            tau = (self.l**2 * (1-self.p_dropout)) / (2 * self.N * self.weight_decay)

            y_var = 1 / tau + torch.var(res, axis=0)
            return y_mean, y_var
        else:
            return self.layers(x)
