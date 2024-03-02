import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, x_list, device):
        self.n_datasets = len(x_list)
        self.datasets = []
        for x in x_list:
            if x.dtype == bool:
                self.datasets.append(
                    torch.from_numpy(x).to(device)
                )
            else:
                self.datasets.append(
                    torch.Tensor(x).to(device)
                )

    def __len__(self):
        return self.datasets[0].shape[0]

    def __getitem__(self, idx):
        return [x[idx] for x in self.datasets]


