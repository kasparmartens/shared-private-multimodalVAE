import torch
import torch.nn as nn
import torchvision
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torch.distributions import kl_divergence
import math

import numpy as np
import pandas as pd


def KL_qp_like(q):
    p = Normal(loc=torch.zeros_like(q.loc), scale=torch.ones_like(q.scale))
    KL_qp = kl_divergence(q, p)
    return KL_qp

def subset_normal(p, column_idx):
    return Normal(loc=p.loc[:, column_idx], scale=p.scale[:, column_idx])
