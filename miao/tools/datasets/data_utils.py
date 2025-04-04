import numpy as np
from torch.utils.data import Dataset, Subset
import torch
import os
import pickle
import polars as pl
import pandas as pd

class DatasetTemplate(Dataset):
    def __init__(self, dataset_cfg=None, root_path=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.root_path = root_path

    def loaddata(self):
        raise NotImplementedError
    
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        raise NotImplementedError
    
    # get indexes for train and test rows
    def get_splits(self, n_test=0.3, n_val=0.2):
        # determine sizes
        total_length = len(self.X)
        test_size = round(n_test * total_length)
        val_size = round(n_val * total_length)
        train_size = total_length- test_size - val_size
        # calculate the split
        # return random_split(self, [train_size, test_size])
        indices = np.arange(len(self.X))

        train_set = Subset(self, indices[ : train_size])  
        val_set = Subset(self, indices[train_size : train_size + val_size])
        test_set = Subset(self, indices[train_size + val_size: ])
        return [train_set, val_set, test_set]



class CSDataset(DatasetTemplate):
    '''
    简单截面数据，每个symbol只含有一个特征。
    X.shape = (n_timestamp, n_symbols)
    '''
    def loaddata(self, dataset_path, device):
        X = pd.read_parquet(os.path.join(dataset_path, "X.parquet")).values
        y = pd.read_parquet(os.path.join(dataset_path, "y.parquet")).values
        with open(os.path.join(dataset_path, "extra_data.pkl"), 'rb') as f:
            extra_data = pickle.load(f)
        assert len(X) == len(y), "length of X and y must be the same"
        # store the inputs and outputs
        self.X = torch.tensor(X, device=device, dtype=torch.float32)
        self.y = torch.tensor(y, device=device, dtype=torch.float32)
        self.mask = torch.ones_like(self.X, device=device)
        self.extra_data = extra_data

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.mask[idx]]
    
    def get_extra_data(self):
        '''获取la_df数据行列，用于还原数据'''
        dt = self.extra_data['dt']
        symbols = self.extra_data['symbols']
        return dt, symbols


class MultiFeatsDataset(DatasetTemplate):
    # load the dataset
    def loaddata(self, dataset_path, device):
        with open(os.path.join(dataset_path, "cs_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "la15_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "extra_data.pkl"), 'rb') as f:
            extra_data = pickle.load(f)
        
        eod_data = torch.tensor(eod_data, device=device, dtype=torch.float32)
        mask_data = torch.tensor(mask_data, device=device, dtype=torch.float32)
        gt_data = torch.tensor(gt_data, device=device, dtype=torch.float32)
        eod_data = eod_data.permute(1, 0, 2)
        mask_data, gt_data = [x.permute(1, 0) for x in [mask_data, gt_data]]

        assert len(eod_data) == len(gt_data), "length of X and y must be the same"
        assert len(eod_data) == len(mask_data), "length of eod_data and mask must be the same"
        # store the inputs and outputs
        self.X = eod_data
        self.y = gt_data
        self.mask = mask_data
        self.extra_data = extra_data

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.mask[idx]]
    