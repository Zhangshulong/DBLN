import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def normalize_train_val_test(train_set, val_set, test_set):
    mean = train_set.mean()
    std = train_set.std()
    
    return (train_set - mean) / std, (val_set - mean) / std, (test_set - mean) / std

def fill_nan_value(train_set, val_set, test_set):
    # Compute per-column mean robustly without triggering "Mean of empty slice" warnings.
    # Use nansum and valid counts so columns with all-NaN get a safe default value.
    sums = np.nansum(train_set, axis=0)
    counts = np.sum(~np.isnan(train_set), axis=0)
    # Where count is zero (all values are NaN), set mean to a small constant to avoid zeros/NaNs.
    # Use np.divide with out/where to avoid evaluating sums/counts division where counts == 0
    col_mean = np.divide(sums, counts, out=np.full_like(sums, 1e-6, dtype=np.float64), where=counts>0)

    # Fill NaNs in train/val/test using the computed column means (no-op if no NaNs)
    ind = np.where(np.isnan(train_set))
    if ind[0].size > 0:
        train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    if ind_val[0].size > 0:
        val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    if ind_test[0].size > 0:
        test_set[ind_test] = np.take(col_mean, ind_test[1])

    return train_set, val_set, test_set

def preprocess_data(sum_dataset, sum_target, seed):
    train_set, val_set, train_target, val_target = train_test_split(sum_dataset, sum_target, test_size = 0.20, random_state = seed)
    train_set, test_set, train_target, test_target = train_test_split(train_set, train_target, test_size = 0.25, random_state = seed)
    # [b, c, t] -> [b, t, c]
    train_set = np.transpose(train_set, (0, 2, 1))
    val_set = np.transpose(val_set, (0, 2, 1))
    test_set = np.transpose(test_set, (0, 2, 1))

    train_set, val_set, test_set = fill_nan_value(train_set, val_set, test_set)

    train_set, val_set, test_set = normalize_train_val_test(train_set, val_set, test_set)
    return train_set, val_set, test_set, train_target, val_target, test_target