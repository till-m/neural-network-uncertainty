import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import torch
from torch.utils.data import Dataset, DataLoader


def func(x):
    return x + 0.5 * np.sin(2*np.pi*(x)) + 0.5 * np.sin(4 * np.pi * (x))


def generate_data(n=2_000, seed=42, exclude=(-0.7, 0.7)):
    # from https://gdmarmerola.github.io/risk-and-uncertainty-deep-learning/
    state = RandomState(MT19937(SeedSequence(seed)))

    x = []
    while len(x) < n:
        candidates = state.normal(loc=0.0, scale=1.0, size=n)
        candidates = candidates[(candidates <= exclude[0]) | (candidates >= exclude[1])]
        x.extend(candidates)
    x = np.array(x[:n])
    eps = state.normal(loc=0.0, scale=1.0, size=n) * (0.01  + 0.2 * np.maximum(0, np.abs(x)-exclude[1])**2)
    x_err = x + eps
    y = func(x_err)
    return x, y

def add_x_noise(x, seed=None):
    if seed is not None:
        seed = seed * 42
        state = RandomState(MT19937(SeedSequence(seed)))
    else:
        state = RandomState()
    eps = state.normal(loc=0.0, scale=1.0, size=x.shape) * (0.01  + 0.1 * x**2)
    return x + eps

class MyDataset(Dataset):
    def __init__(self, generator_kwargs=dict()) -> None:
        super().__init__()
        x, y = generate_data(**generator_kwargs)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[[idx]], self.y[[idx]]

class MyDataloader(DataLoader):
    def __init__(self, dataset_kwargs=dict(), **kwargs):
        super().__init__(MyDataset(**dataset_kwargs), **kwargs)
