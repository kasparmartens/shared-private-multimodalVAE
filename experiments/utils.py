import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def generate_missingness_masks(n, frac_x1_missing, frac_x2_missing):
    mask_x1 = torch.zeros(n, 1).bool()
    mask_x2 = torch.zeros(n, 1).bool()
    mask_y = torch.zeros(n, 1).bool()
    n_obs1 = int(frac_x1_missing * n)
    n_obs2 = int(frac_x2_missing * n)
    frac_both = 1.0 - frac_x1_missing - frac_x2_missing
    n_obs_both = int(frac_both * n)
    mask_x2[0:n_obs1] = True
    mask_x1[(n_obs1 + n_obs_both):] = True
    # shuffle the rows of mask_x1
    mask_x1 = mask_x1[torch.randperm(n)]
    mask_x2 = mask_x2[torch.randperm(n)]
    return mask_x1, mask_x2, mask_y

def train_test_val_split(x1, x2, y, test_size=0.2, val_size=0.2, random_state=0):
    # first split into (train + val) and test
    x1_train_and_val, x1_test, x2_train_and_val, x2_test, y_train_and_val, y_test = train_test_split(x1, x2, y, stratify=y, test_size=test_size, random_state=random_state)
    # now split further into train and val
    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train_and_val, x2_train_and_val, y_train_and_val, stratify=y_train_and_val, test_size=val_size, random_state=random_state)
    return x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, y_train, y_val, y_test

