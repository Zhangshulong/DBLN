import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preprocess.data_load_uea import uea_load_data
from data_preprocess.base_loader import base_loader
from data_preprocess.data_preprocess import preprocess_data

def load_data(root_path, datatype, dataset,):
    dataroot = root_path + '/' + datatype
    if datatype == 'UEA':
        sum_dataset, sum_target, num_classes = uea_load_data(dataroot, dataset)
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    return sum_dataset, sum_target, num_classes

def data_loaders(batch_size, sum_dataset, sum_target, seed):
    train_set, val_set, test_set, train_target, val_target, test_target = preprocess_data(sum_dataset, sum_target, seed)
    train_set = base_loader(torch.from_numpy(train_set).type(torch.FloatTensor), torch.from_numpy(train_target).type(torch.FloatTensor).to(torch.int64))
    val_set = base_loader(torch.from_numpy(val_set).type(torch.FloatTensor), torch.from_numpy(val_target).type(torch.FloatTensor).to(torch.int64))
    test_set = base_loader(torch.from_numpy(test_set).type(torch.FloatTensor), torch.from_numpy(test_target).type(torch.FloatTensor).to(torch.int64))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader