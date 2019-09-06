#!/usr/bin/env python36
# -*- coding:utf-8 -*-
# @Time    : 19-8-9 下午2:52
# @Author  : Xinxin Zhang
import torch
import torch.utils.data as Data


class Pack(dict):
    def __getattr__(self, name):
        return self.get(name)

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack


def list2tensor(X):
    """
    list2tensor
    """
    size = maxLens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            ll = len(x)
            tensor[i, :ll] = torch.tensor(x)
            lengths[i] = ll
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                ll = len(x)
                tensor[i, j, :ll] = torch.tensor(x)
                lengths[i, j] = ll

    return tensor, lengths


def maxLens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [
            len(X),
            max(len(x) for x in X),
            max(len(x) for xs in X for x in xs)
        ]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


class Dataloader(Data.Dataset):
    """
    dataloader
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch

        return collate

    def createBatches(self, batch_size=1, shuffle=False, device=0):
        """
        create_batches
        """
        loader = Data.DataLoader(dataset=self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=self.collate_fn(device),
                                 pin_memory=False)
        return loader
