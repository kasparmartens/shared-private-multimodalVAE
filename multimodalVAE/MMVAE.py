import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.bernoulli import Bernoulli

from torch.distributions import kl_divergence
import math

from .eval_utils import calculate_Rsq, corr
from .linear_classifier import helper_fit_logreg
from .distributions_utils import myPoisson, myCategorical


class MMVAE(nn.Module):
    def __init__(self, data_dims, z_dim, sparsity_mask, encoder_list, decoder_list, cross_view_modification, use_labels=False, likelihood="Bernoulli", lr=1e-3, device="cpu"):
        super().__init__()

        self.n_views = len(data_dims)
        self.data_dims = data_dims
        self.sparsity_mask = sparsity_mask
        if isinstance(likelihood, list):
            self.likelihoods = likelihood
        else:
            self.likelihoods = [likelihood for _ in range(self.n_views)]

        self.noise_sd = nn.ParameterList(
            [nn.Parameter(-1 * torch.ones(1, data_dim)) if likelihood == "Gaussian" else None for (data_dim, likelihood) in zip(self.data_dims, self.likelihoods)]
        )

        # setting up VAE
        self.z_dim = z_dim
        self.encoder_list = nn.ModuleList(encoder_list)
        self.decoder_list = nn.ModuleList(decoder_list)

        self.use_labels = use_labels
        self.cross_view_modification = cross_view_modification

        self.device = device
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def _get_likelihood_distr(self, params, m):
        likelihood = self.likelihoods[m]
        if likelihood == "Gaussian":
            p_x = Normal(loc=params, scale=F.softplus(self.noise_sd[m]))
        elif likelihood == "Bernoulli":
            p_x = Bernoulli(logits=params, validate_args=False)
        elif likelihood == "Poisson":
            p_x = myPoisson(rate=params)
        elif likelihood == "Categorical":
            p_x = myCategorical(logits=params)
        return p_x

    def _get_encoding_distr(self, encoder_output):
        mu_z, log_sigma_z = torch.split(encoder_output, self.z_dim, dim=1)
        return Normal(mu_z, F.softplus(log_sigma_z))

    def get_average_z(self, qz_list):
        assert self.n_views == 2
        with torch.no_grad():
            qz1 = qz_list[0].mean
            qz2 = qz_list[1].mean
            z = torch.zeros_like(qz_list[0].mean)
            # sparsity mask, assume two views
            s = (self.get_sparsity_mask(sample=False) + 0.5).cpu().int().numpy().astype(np.bool)
            # z1_private
            which_dims_private1 = np.bitwise_and(s[0], ~s[1])
            z[:, which_dims_private1] = qz1[:, which_dims_private1]
            # z_shared
            which_dims_shared = np.bitwise_and(s[0], s[1])
            z[:, which_dims_shared] = 0.5 * (qz1[:, which_dims_shared] + qz2[:, which_dims_shared])
            # z2_private
            which_dims_private2 = np.bitwise_and(~s[0], s[1])
            z[:, which_dims_private2] = qz2[:, which_dims_private2]
            # n_contributions = s.sum(dim=0, keepdims=True)
            # for m in range(self.n_views):
            #     which_z_dims = s[m]
            #     z[:, which_z_dims] = qz_list[m].mean[:, which_z_dims]
            # z /= n_contributions
        return z
    
    def get_sparsity_mask(self, sample=False):
        return self.sparsity_mask

    def encode(self, x_list):
        qz_list = []
        for m in range(self.n_views):
            qz_list.append(
                self._get_encoding_distr(
                    self.encoder_list[m](x_list[m])
                )
            )
        return qz_list

    def decode(self, z, s):
        px_z_list = []
        for m in range(self.n_views if not self.use_labels else self.n_views + 1):
            z_active = s[m] * z
            pred = self.decoder_list[m](z_active)
            px_z = self._get_likelihood_distr(pred, m)
            px_z_list.append(px_z)
        return px_z_list

    def forward(self, x_list, use_posterior_mean=False):
        qz_list = self.encode(x_list)
        pred_list_list = []
        s = self.get_sparsity_mask(sample=True)
        for j in range(len(qz_list)):
            if use_posterior_mean:
                z = qz_list[j].mean
            else:
                z = qz_list[j].rsample()
            # private vs shared coordinates encoding
            z = z * s[j:(j+1)]
            # p(x^{1:M} | z) for z ~ q_{j}(z)
            px_z_list = self.decode(z, s)
            pred_list_list.append(px_z_list)
        return pred_list_list

    def get_z_shared_and_private(self, x_list):
        qz_list = self.encode(x_list)
        z_avg = self.get_average_z(qz_list)
        z1 = qz_list[0].mean
        z2 = qz_list[1].mean
        s = self.get_sparsity_mask(sample=True)
        s_binary = (s + 0.5).cpu().int().numpy().astype(np.bool)
        which_dims_shared = np.bitwise_and(s_binary[0], s_binary[1])
        which_dims_private1 = np.bitwise_and(s_binary[0], ~s_binary[1])
        which_dims_private2 = np.bitwise_and(~s_binary[0], s_binary[1])
        return z_avg[:, which_dims_shared], z1[:, which_dims_private1], z2[:, which_dims_private2]


    def forward_and_loss(self, x_list, x_mask_list, modality_weights, beta=1.0, use_posterior_mean=False, cross_view_terms_only=False):
        qz_list = self.encode(x_list)
        KL = 0.0
        logliks = 0.0
        s = self.get_sparsity_mask(sample=True)
        # define latent space split [z1_private, z_shared, z2_private]
        s0 = (self.get_sparsity_mask() + 0.5).cpu().int().numpy().astype(np.bool)
        which_dims_shared = np.bitwise_and(s0[0], s0[1])
        px_z_list = []
        for j in range(len(qz_list)):
            if use_posterior_mean:
                z = qz_list[j].mean
            else:
                z = qz_list[j].rsample()
            # zero out private/shared coordinates of z
            if j < self.n_views:
                z = z * s[j:(j+1)]
            # p(x^{1:M} | z) for z ~ q_{j}(z)
            # px_z_list = self.decode(z, s)
            # KL
            p_normal = Normal(torch.zeros_like(z), torch.ones_like(z))
            kl_term = kl_divergence(qz_list[j], p_normal) * s[j:(j+1)]
            # mask out missing data points from modality j, kl_term has shape [batch_size, n_features]
            KL += kl_term.masked_fill(x_mask_list[j], 0.0).sum(dim=1).mean(dim=0)
            # modified gradient
            for m in range(self.n_views if not self.use_labels else self.n_views + 1):
                if self.cross_view_modification:
                    # is it cross-view pred? this is normal forward pass
                    if (j == 0) & (m == 1) or (j == 1) & (m == 0) or (m == 2):
                        # if yes, then allow gradients
                        z_active = s[m] * z
                        pred = self.decoder_list[m](z_active)
                        px_m = self._get_likelihood_distr(pred, m)
                        log_p = px_m.log_prob(x_list[m]).masked_fill(x_mask_list[j] | x_mask_list[m], 0.0)
                        logliks += modality_weights[m] * log_p.sum(dim=1).mean(dim=0)
                    else:
                        # for same-view pred, allow using values from z_shared but cut shared gradients
                        z_copy = torch.zeros_like(z)
                        z_copy[:, which_dims_shared] = z[:, which_dims_shared].detach()
                        z_copy[:, ~which_dims_shared] = z[:, ~which_dims_shared]
                        z_active = s[m] * z_copy
                        pred = self.decoder_list[m](z_active)
                        px_m = self._get_likelihood_distr(pred, m)
                        log_p = px_m.log_prob(x_list[m]).masked_fill(x_mask_list[j] | x_mask_list[m], 0.0)
                        if len(x_list[m].shape) == 2:
                            logliks += modality_weights[m] * log_p.sum(dim=1).mean(dim=0)
                        else:
                            logliks += modality_weights[m] * log_p.sum(dim=[1, 2, 3]).mean(dim=0)
                else:
                    z_active = s[m] * z
                    pred = self.decoder_list[m](z_active)
                    px_m = self._get_likelihood_distr(pred, m)
                    log_p = px_m.log_prob(x_list[m]).masked_fill(x_mask_list[j] | x_mask_list[m], 0.0)
                    logliks += modality_weights[m] * log_p.sum(dim=1).mean(dim=0)
        total_loss = -1.0 * logliks + beta * KL
        return total_loss

    def step(self, x_list, x_mask_list, modality_weights=None, beta=1.0, cross_view_terms_only=False):
        if modality_weights is None:
            modality_weights = np.ones(len(x_list))
        loss = self.forward_and_loss(x_list, x_mask_list, modality_weights, beta=beta, cross_view_terms_only=cross_view_terms_only)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def quantify_cross_view_pred(self, x1, x2):
        self.eval()
        pred_list_list = self.forward([x1, x2], use_posterior_mean=True)
        self.train()
        self_pred1 = pred_list_list[0][0].mean
        self_pred2 = pred_list_list[1][1].mean
        cross_pred1 = pred_list_list[1][0].mean
        cross_pred2 = pred_list_list[0][1].mean
        joint_pred1 = 0.5 * (self_pred1 + cross_pred1)
        joint_pred2 = 0.5 * (self_pred2 + cross_pred2)
        Rsq_cross1 = calculate_Rsq(x1, cross_pred1)
        Rsq_self1 = calculate_Rsq(x1, self_pred1)
        Rsq_cross2 = calculate_Rsq(x2, cross_pred2)
        Rsq_self2 = calculate_Rsq(x2, self_pred2)
        Rsq_joint1 = calculate_Rsq(x1, joint_pred1)
        Rsq_joint2 = calculate_Rsq(x2, joint_pred2)
        df1 = pd.DataFrame({"view": 1, "feature": np.arange(x1.shape[1]), "Rsq_cross": Rsq_cross1, "Rsq_self": Rsq_self1, "Rsq_joint": Rsq_joint1})
        df2 = pd.DataFrame({"view": 2, "feature": np.arange(x2.shape[1]), "Rsq_cross": Rsq_cross2, "Rsq_self": Rsq_self2, "Rsq_joint": Rsq_joint2})
        df_Rsq = pd.concat([df1, df2])
        return df_Rsq

    def quantify_separation_in_latent_space(self, x_train_list, y_train, x_test_list, y_test):
        with torch.no_grad():
            z_train = self.get_average_z(
                self.encode(x_train_list)
            ).cpu().numpy()
            z_test = self.get_average_z(
                self.encode(x_test_list)
            ).cpu().numpy()
            # sparsity mask, assume two views
            s = (self.get_sparsity_mask() + 0.5).cpu().int().numpy().astype(np.bool)
            # z1_private
            which_dims_private1 = np.bitwise_and(s[0], ~s[1])
            acc1, auc1 = helper_fit_logreg(z_train[:, which_dims_private1], y_train, z_test[:, which_dims_private1], y_test)
            # z_shared
            which_dims_shared = np.bitwise_and(s[0], s[1])
            acc2, auc2 = helper_fit_logreg(z_train[:, which_dims_shared], y_train, z_test[:, which_dims_shared], y_test)
            # z2_private
            which_dims_private2 = np.bitwise_and(~s[0], s[1])
            acc3, auc3 = helper_fit_logreg(z_train[:, which_dims_private2], y_train, z_test[:, which_dims_private2], y_test)
            # overall
            acc4, auc4 = helper_fit_logreg(z_train, y_train, z_test, y_test)
            # true_class = df_label.er_status.values
            return pd.DataFrame({
                "latent_dims": ["private1", "shared", "private2", "overall"],
                "accuracy": [acc1, acc2, acc3, acc4], 
                "AUC": [auc1, auc2, auc3, auc4]
            })

    def correlate_latent_coords_with_features(self, x_train_list, x_test_list):
        with torch.no_grad():
            # on the training set
            z_train = self.get_average_z(
                self.encode(x_train_list)
            ).cpu().numpy()
            corr_z_view1_train = corr(z_train, x_train_list[0].cpu().numpy())
            corr_z_view2_train = corr(z_train, x_train_list[1].cpu().numpy())
            # on the test set
            z_test = self.get_average_z(
                self.encode(x_test_list)
            ).cpu().numpy()
            corr_z_view1_test = corr(z_test, x_test_list[0].cpu().numpy())
            corr_z_view2_test = corr(z_test, x_test_list[1].cpu().numpy())
        return corr_z_view1_train, corr_z_view2_train, corr_z_view1_test, corr_z_view2_test

    def extract_latent_coords(self, x_list):
        with torch.no_grad():
            qz_list = self.encode(x_list)
            z_avg = self.get_average_z(qz_list).cpu().numpy()
        return z_avg
    