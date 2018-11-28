from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import sys
import argparse
import json


class DMDataset(data.Dataset):
        def __init__(self, seq_len=10, size=20, dataset_size=1000):
            self.seq_len = seq_len
            self.size = size
            self.dataset_size = dataset_size
            self.x_cache = np.random.rand(self.seq_len, self.size, self.size)

        def __getitem__(self, index):
            idx = index % self.seq_len
            x = self.x_cache[range(idx,self.seq_len)+range(0,idx)]
            target = self.x_cache[idx]
            return x, target

        def __len__(self):
            return self.dataset_size
