import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent experts.
    
    Adapted from https://github.com/mhw32/multimodal-vae-public/blob/master/celeba/model.py
    """
    def __init__(self, include_prior_expert=True):
        super().__init__()
        self.include_prior_expert = include_prior_expert

    def forward(self, q_list, x_mask_list, s, eps=1e-8):
        mu_list = []
        prec_list = []
        if self.include_prior_expert:
            # initialise with prior experts
            mu_list.append(
                torch.zeros_like(q_list[0].loc[None])
            )
            prec_list.append(
                torch.ones_like(q_list[0].scale[None])
            )
        for j in range(len(q_list)):
            q = q_list[j]
            mu_list.append(
                (q.loc[None] * s[j, :]).masked_fill(x_mask_list[j], 0)
            )
            prec_list.append(
                (1.0 / (q.scale[None] ** 2 + eps) * s[j, :]).masked_fill(x_mask_list[j], 0)
            )
        # shape [n_modalities+1, batch_size, latent_dim]
        mu = torch.cat(mu_list, dim=0)
        # shape [n_modalities+1, batch_size, latent_dim]
        precision = torch.cat(prec_list, dim=0)
        product_mu = torch.sum(mu * precision, dim=0) / torch.sum(precision, dim=0)
        product_var = 1. / torch.sum(precision, dim=0)
        PoE = Normal(loc=product_mu, scale=torch.sqrt(product_var))
        return PoE
