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
        def __init__(self, cyc_len=10,seq_len=10, dm_size=20, dataset_size=1000, 
                gen_rule="gravity-sparse", p=1, train=True, seq_num=3):
            self.cyc_len = cyc_len
            self.seq_len = seq_len
            self.size = dm_size
            self.dataset_size = dataset_size
            self.gen_rule = gen_rule
            self.p = p
            self.train = train
            self.seq_num = seq_num

            print("Is_train:", train, "gen_rule:", gen_rule)
            if "gravity" in gen_rule:
#                A = np.random.RandomState(seed=100).rand(self.size,1)
#                Rf = np.random.RandomState(seed=101).rand(self.size, self.size)
#                self.x = A * Rf
#                print(self.x)
                if "cycle" in gen_rule:
                    self.gen_seq(self.cyc_len)
                    self.x_cache = np.tile(self.x_cyc,(1,self.seq_len//self.cyc_len+2,1,1))
                elif "avg" in gen_rule:
                    self.gen_seq(self.seq_len)
                    self.x_cache = np.zeros((self.seq_num,self.seq_len+self.dataset_size, self.size, self.size))
                    self.x_cache[:,:self.seq_len] = self.x_cyc
                    for i_seq in range(self.seq_len, self.seq_len+self.dataset_size):
                        self.x_cache[:,i_seq] = (self.x_cache[:,(i_seq - self.seq_len):i_seq]).mean((1))
                elif "random" in gen_rule:
                    self.x_cache = self.gen_seq(self.seq_len*2)
                    
                else:
                    assert 0
                #    self.x_cache =
                #    np.random.RandomState(seed=102).rand(self.cyc_len, self.size,self.size)
            else:
                assert 0
                self.x_cache = np.random.rand(self.cyc_len, self.size, self.size)

        def gen_seq_invariant_gravity(self,seq_len):
            self.x_cyc = np.zeros((seq_len, self.size, self.size))
            for i_seq in range(seq_len):
                self.x_cyc[i_seq] = self.x
                mask = (np.random.RandomState(seed=102*i_seq).rand(self.size,self.size)<self.p)
                self.x_cyc[i_seq] = self.x_cyc[i_seq] * mask
            return self.x_cyc

        
        def gen_seq(self,seq_len):
            A = np.random.RandomState(seed=100+self.train).rand(self.seq_num,seq_len,self.size,1)
            Rf = np.random.RandomState(seed=102+self.train).rand(self.seq_num,seq_len,self.size, self.size)
            self.x_cyc = A * Rf
            mask = (np.random.RandomState(seed=104+self.train).rand(self.seq_num,seq_len,self.size,self.size)<self.p)
            self.x_cyc = self.x_cyc * mask
            return self.x_cyc

        def __getitem__(self, index):
            #x = np.zeros((self.seq_len, self.size, self.size))
            #print(self.x_cache[:,:5,:5])
            seq_idx = index % self.seq_num
            if "cycle" in self.gen_rule:
                idx = index % self.cyc_len
#                x_perm = self.x_cache[range(idx,self.cyc_len)+range(0,idx)]
#                for i_cyc in range(self.seq_len//self.cyc_len):
#                    x[i_cyc*self.cyc_len:((i_cyc+1)*self.cyc_len)] = x_perm
#                x[i_cyc*self.cyc_len:] = x_perm[:self.seq_len-(i_cyc*self.cyc_len)]
                x = self.x_cache[seq_idx,idx:(idx+self.seq_len)]
                target = self.x_cache[seq_idx,(idx+self.seq_len)]
            elif "avg" in self.gen_rule:
                x = self.x_cache[seq_idx, index:(index+self.seq_len)]
                target = self.x_cache[seq_idx,index+self.seq_len]
            elif "random" in self.gen_rule:
                perm = np.random.RandomState(seed=index).choice(self.seq_len*2,self.seq_len+1)
                x = self.x_cache[seq_idx,perm[:-1]]
                target = self.x_cache[seq_idx,perm[-1]]

#                for i_seq in range(self.seq_len):
#                    if self.train:
#                        mask = (np.random.RandomState(seed=index*i_seq+self.dataset_size*self.seq_len).rand(self.size,self.size)<self.p)
#                    else:
#                        mask = (np.random.RandomState(seed=index*i_seq).rand(self.size,self.size)<self.p)
#                    x[i_seq] = self.x_cache[i_seq] * mask
#                if self.train:
#                    mask = (np.random.RandomState(seed=index*(1+i_seq)+self.dataset_size*self.seq_len).rand(self.size,self.size)<self.p)
#                else:
#                    mask = (np.random.RandomState(seed=index*(1+i_seq)).rand(self.size,self.size)<self.p)
#                target = self.gen_seq(1)[0] * mask
            return x, target

        def __len__(self):
            return self.dataset_size
