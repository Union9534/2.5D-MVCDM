from torch.utils.data import Dataset
import torch
import os
import scipy.io as sio
import numpy as np

class SeisDataset(Dataset):
    def __init__(
        self,
        gt_data
    ):
        super().__init__()
        self.gt_data = gt_data

    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, idx):
        gt_data = self.gt_data[idx]
        return gt_data
