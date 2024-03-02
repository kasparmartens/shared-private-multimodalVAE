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
from torch.distributions.categorical import Categorical

from torch.distributions import kl_divergence
import math

from .linear_classifier import helper_fit_logreg
from .distributions_utils import myPoisson, myCategorical


from .PoE import ProductOfExperts
from .utils import subset_normal, KL_qp_like
from .eval_utils import calculate_Rsq

class MVAE(nn.Module):
    def __init__(self, data_dims, z_dim, sparsity_mask, encoder_list, decoder_list, use_labels=False, learn_sparsity_mask=False, include_cross_view_terms=False, likelihood="Bernoulli", lr=1e-3, device="cpu"):
        super().__init__()

        self.n_views = len(data_dims)
        self.data_dims = data_dims
        self.sparsity_mask = sparsity_mask
        s = sparsity_mask.cpu().int().numpy().astype(np.bool)
        # z1_private
        self.which_dims_private1 = np.bitwise_and(s[0], ~s[1])
        # z_shared
        self.which_dims_shared = np.bitwise_and(s[0], s[1])
        # z2_private
        self.which_dims_private2 = np.bitwise_and(~s[0], s[1])
        # should we use the modified ELBO?
        self.include_cross_view_terms = include_cross_view_terms

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
        self.learn_sparsity_mask = learn_sparsity_mask

        self.use_labels = use_labels
        self.PoE = ProductOfExperts()

        self.device = device
        self.optim = torch.optim.Adam(self.parameters(), lr)

    def get_sparsity_mask(self, sample=False):
        return self.sparsity_mask

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
        return Normal(mu_z, F.softplus(log_sigma_z), validate_args=False)

    def get_average_z(self, qz_list):
        assert self.n_views == 2
        with torch.no_grad():
            qz1 = qz_list[0].mean
            qz2 = qz_list[1].mean
            z = torch.zeros_like(qz_list[0].mean)
            # sparsity mask, assume two views
            s = self.get_sparsity_mask(sample=False).cpu().int().numpy().astype(np.bool)
            # z1_private
            which_dims_private1 = np.bitwise_and(s[0], ~s[1])
            z[:, which_dims_private1] = qz1[:, which_dims_private1]
            # z_shared
            which_dims_shared = np.bitwise_and(s[0], s[1])
            z[:, which_dims_shared] = 0.5 * (qz1[:, which_dims_shared] + qz2[:, which_dims_shared])
            # z2_private
            which_dims_private2 = np.bitwise_and(~s[0], s[1])
            z[:, which_dims_private2] = qz2[:, which_dims_private2]
        return z

    def encode(self, x_list):
        qz_list = []
        for m in range(self.n_views):
            qz_list.append(
                self._get_encoding_distr(
                    self.encoder_list[m](x_list[m])
                )
            )
        return qz_list

    def decode(self, z, sample_binary_mask=True, apply_last_layer_sigmoid=False):
        px_z_list = []
        s = self.get_sparsity_mask(sample_binary_mask)
        for m in range(self.n_views if not self.use_labels else self.n_views + 1):
            z_active = s[m] * z
            pred = self.decoder_list[m](z_active)
            px_z = self._get_likelihood_distr(pred, m)
            px_z_list.append(px_z)
        return px_z_list

    def forward_and_loss(self, x_list, x_mask_list, modality_weights, beta=1.0, use_posterior_mean=False):
        qz_list = self.encode(x_list)
        # initialise loss
        KL = 0.0
        logliks = 0.0
        # z_both
        qz = self.PoE(qz_list[0:2], x_mask_list[0:2], self.sparsity_mask[0:2])
        z = qz.mean if use_posterior_mean else qz.rsample()
        px_z_list = self.decode(z)
        KL += KL_qp_like(qz).sum(dim=1).mean(dim=0)
        for m in range(self.n_views if not self.use_labels else self.n_views + 1):
            # p(x^m | z) for z ~ q_j(z)
            log_p = px_z_list[m].log_prob(x_list[m]).masked_fill(x_mask_list[m], 0.0)
            logliks += modality_weights[m] * log_p.sum(dim=1).mean(dim=0)
        # z_private1
        qz = self.PoE(qz_list[0:1], x_mask_list[0:1], self.sparsity_mask[0:1])
        z = qz.mean if use_posterior_mean else qz.rsample()
        z[:, self.which_dims_private2].fill_(0.0)
        px_z_list = self.decode(z)
        KL += KL_qp_like(
            # take all coordinates except z_private2
            subset_normal(qz, ~self.which_dims_private2)
        ).sum(dim=1).mean(dim=0)
        log_p = px_z_list[0].log_prob(x_list[0]).masked_fill(x_mask_list[0], 0.0)
        logliks += modality_weights[0] * log_p.sum(dim=1).mean(dim=0)
        # classifier loglik
        if self.use_labels:
            logliks += modality_weights[2] * px_z_list[2].log_prob(x_list[2]).sum(dim=1).mean(dim=0)
        if self.include_cross_view_terms:
            qz = self.PoE(qz_list[0:1], x_mask_list[0:1], self.sparsity_mask[0:1])
            z = qz.mean if use_posterior_mean else qz.rsample()
            z[:, ~self.which_dims_shared].fill_(0.0)
            px_z_list = self.decode(z)
            KL += KL_qp_like(
                # take only shared coordinates
                subset_normal(qz, self.which_dims_shared)
            ).sum(dim=1).mean(dim=0)
            log_p = px_z_list[1].log_prob(x_list[1]).masked_fill(x_mask_list[1], 0.0)
            logliks += modality_weights[1] * log_p.sum(dim=1).mean(dim=0)
            # classifier loglik
            if self.use_labels:
                logliks += modality_weights[2] * px_z_list[2].log_prob(x_list[2]).sum(dim=1).mean(dim=0)
        # z_private2
        qz = self.PoE(qz_list[1:2], x_mask_list[1:2], self.sparsity_mask[1:2])
        z = qz.mean if use_posterior_mean else qz.rsample()
        z[:, self.which_dims_private1].fill_(0.0)
        px_z_list = self.decode(z)
        KL += KL_qp_like(
            # take all coordinates except z_private1
            subset_normal(qz, ~self.which_dims_private1)
        ).sum(dim=1).mean(dim=0)
        log_p = px_z_list[1].log_prob(x_list[1]).masked_fill(x_mask_list[1], 0.0)
        logliks += modality_weights[1] * log_p.sum(dim=1).mean(dim=0)
        # classifier loglik
        if self.use_labels:
            logliks += modality_weights[2] * px_z_list[2].log_prob(x_list[2]).sum(dim=1).mean(dim=0)
        if self.include_cross_view_terms:
            qz = self.PoE(qz_list[1:2], x_mask_list[1:2], self.sparsity_mask[1:2])
            z = qz.mean if use_posterior_mean else qz.rsample()
            z[:, ~self.which_dims_shared].fill_(0.0)
            px_z_list = self.decode(z)
            KL += KL_qp_like(
                # take only shared coordinates
                subset_normal(qz, self.which_dims_shared)
            ).sum(dim=1).mean(dim=0)
            log_p = px_z_list[0].log_prob(x_list[0]).masked_fill(x_mask_list[0], 0.0)
            logliks += modality_weights[0] * log_p.sum(dim=1).mean(dim=0)
            # classifier loglik
            if self.use_labels:
                logliks += modality_weights[2] * px_z_list[2].log_prob(x_list[2]).sum(dim=1).mean(dim=0)
        # total loss
        total_loss = -1.0 * logliks + beta * KL
        return total_loss

    def step(self, x_list, x_mask_list, modality_weights=None, beta=1.0):
        if modality_weights is None:
            modality_weights = np.ones(len(x_list))
        loss = self.forward_and_loss(x_list, x_mask_list, modality_weights, beta=beta)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def get_joint_preds(self, x1, x2):
        qz_list = self.encode([x1, x2])
        s = self.get_sparsity_mask()
        # use a dummy mask
        dummy_mask = torch.zeros(x1.shape[0], 1).bool().to(self.device)
        qz = self.PoE(qz_list, [dummy_mask, dummy_mask], s)
        # posterior mean
        z = qz.mean
        px_z_list = self.decode(z)
        joint_pred1, joint_pred2 = px_z_list[0].mean, px_z_list[1].mean
        return joint_pred1, joint_pred2


    def get_cross_view_preds(self, x1, x2):
        qz_list = self.encode([x1, x2])
        s = self.get_sparsity_mask()
        pred_list_list = []
        for j in range(len(qz_list)):
            # here it's easiest to use no masking
            dummy_mask = torch.zeros(x1.shape[0], 1).bool().to(self.device)
            qz = self.PoE(qz_list[j:(j + 1)], list(dummy_mask), s[j:(j + 1)])
            # posterior mean
            z = qz.mean
            if j == 0:
                z[:, ~self.which_dims_private2].fill_(0.0)
            elif j == 1:
                z[:, ~self.which_dims_private1].fill_(0.0)
            px_z_list = self.decode(z)
            pred_list_list.append(px_z_list)
        # extract all components
        self_pred1 = pred_list_list[0][0].mean
        self_pred2 = pred_list_list[1][1].mean
        cross_pred1 = pred_list_list[1][0].mean
        cross_pred2 = pred_list_list[0][1].mean
        return self_pred1, self_pred2, cross_pred1, cross_pred2

    def quantify_cross_view_pred(self, x1, x2):
        self.eval()
        self_pred1, self_pred2, cross_pred1, cross_pred2 = self.get_cross_view_preds(x1, x2)
        joint_pred1, joint_pred2 = self.get_joint_preds(x1, x2)
        self.train()
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
        boolean_false_train = [torch.zeros_like(x).bool()[:, 0:1] for x in x_train_list]
        boolean_false_test = [torch.zeros_like(x).bool()[:, 0:1] for x in x_test_list]
        with torch.no_grad():
            s = self.get_sparsity_mask()
            qz_train = self.PoE(
                self.encode(x_train_list), boolean_false_train, s
            )
            z_train = qz_train.mean.cpu().numpy()
            qz_test = self.PoE(
                self.encode(x_test_list), boolean_false_test, s
            )
            z_test = qz_test.mean.cpu().numpy()
            # sparsity mask, assume two views
            s = self.get_sparsity_mask().cpu().int().numpy().astype(np.bool)
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

    def extract_latent_coords(self, x_list):
        with torch.no_grad():
            qz_list = self.encode(x_list)
            dummy_mask1 = torch.zeros(x_list[0].shape[0], 1).bool().to(self.device)
            dummy_mask2 = torch.zeros(x_list[1].shape[0], 1).bool().to(self.device)
            dummy_mask_list = [dummy_mask1, dummy_mask2]
            qz = self.PoE(qz_list[0:2], dummy_mask_list[0:2], self.sparsity_mask[0:2])
            mu_z = qz.mean.cpu().numpy()
        return mu_z
    