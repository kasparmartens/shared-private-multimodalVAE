import pandas as pd
import numpy as np

import torch

def normalise_columns(x, scale=True):
    if scale:
        return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    else:
        return (x - x.mean(axis=0, keepdims=True))


def concat_top_and_bottom(x, top, bottom):
    ncols = x.shape[1]
    if top is None or bottom is None:
        return x
    x_top = x[:, 0:top]
    x_bottom = x[:, (ncols-bottom):]
    return np.hstack((x_top, x_bottom))

