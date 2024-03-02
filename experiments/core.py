import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wandb

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split
from multimodalVAE.MMVAE import MMVAE
from multimodalVAE.MVAE import MVAE
from multimodalVAE.latent_structure_utils import get_latent_structure

from torch.utils.data import DataLoader
from dataset.multiview_dataset import MultiViewDataset
from utils import generate_missingness_masks

def train_and_eval(x1, x2, y, model, latent_structure, supervised, device, modality_weights, n_epochs=100, lr=1e-4, seed=0, wandb_logging=True):
    idx = np.arange(x1.shape[0])
    x1_train, x1_test, x2_train, x2_test, y_train, y_test, idx_train, idx_test = train_test_split(x1, x2, y, idx, test_size=0.2, stratify=y, random_state=seed)
    # missingness masks for training data (in these experiments, we assume that missingness rate is 0.0)
    mask_x1_train, mask_x2_train, mask_y_train = generate_missingness_masks(x1_train.shape[0], frac_x1_missing=0.0, frac_x2_missing=0.0)
    # latent space structure (incl private and shared dimensions)
    latent_mask = get_latent_structure(latent_structure).to(device)
    z_dim = latent_mask.shape[1]
    # data dims
    data_dims = [x1.shape[1], x2.shape[1]]
    # encoder architecture
    encoder_list = [
        nn.Sequential(
            nn.Linear(data_dims[0], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2*z_dim)
        ), 
        nn.Sequential(
            nn.Linear(data_dims[1], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2*z_dim)
        )
    ]
    # decoder architecture
    decoder_list = [
        nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, data_dims[0])
        ), 
        nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, data_dims[1])
        ),
        nn.Sequential(
            nn.Linear(z_dim, 1)
        )
    ]
    likelihoods = ["Gaussian", "Gaussian", "Bernoulli"]
    if model == "MMVAE":
        m = MMVAE(data_dims, z_dim, latent_mask, encoder_list, decoder_list, cross_view_modification=False, use_labels=supervised, likelihood=likelihoods, device=device, lr=lr)
    elif model == "MMVAE++":
        m = MMVAE(data_dims, z_dim, latent_mask, encoder_list, decoder_list, cross_view_modification=True, use_labels=supervised, likelihood=likelihoods, device=device, lr=lr)
    elif model == "MVAE":
        m = MVAE(data_dims, z_dim, latent_mask, encoder_list, decoder_list, include_cross_view_terms=False, use_labels=supervised, likelihood=likelihoods, device=device, lr=lr)
    elif model == "MoPoE-VAE":
        m = MVAE(data_dims, z_dim, latent_mask, encoder_list, decoder_list, include_cross_view_terms=True, use_labels=supervised, likelihood=likelihoods, device=device, lr=lr)
    else:
        raise ValueError(f"Unknown model {model}")
    m.to(device)
    # various datasets for training and evaluation
    mv_dataset = MultiViewDataset([x1_train, mask_x1_train, x2_train, mask_x2_train, y_train, mask_y_train], device=device)
    mv_helper_train = MultiViewDataset([x1_train, x2_train], device=device)
    mv_helper_test = MultiViewDataset([x1_test, x2_test], device=device)
    mv_helper_all = MultiViewDataset([x1, x2], device=device)
    mv_dataset_test = MultiViewDataset([x1_test, x2_test, y_test], device=device)
    # train loader
    data_loader = DataLoader(mv_dataset, batch_size=128, shuffle=True)
    beta_schedule = torch.linspace(10**-5, 1.0, steps=n_epochs)
    df_list = []
    for epoch in range(n_epochs):
        train_loss = 0.0
        for batch_idx, (x1_sub, mask_x1, x2_sub, mask_x2, y_sub, mask_y) in enumerate(data_loader):
            if model == "MMVAE":
                train_loss += m.step([x1_sub, x2_sub, y_sub], [mask_x1, mask_x2, mask_y], modality_weights=modality_weights, beta=beta_schedule[epoch], cross_view_terms_only=False)
            elif model == "MMVAE++":
                train_loss += m.step([x1_sub, x2_sub, y_sub], [mask_x1, mask_x2, mask_y], modality_weights=modality_weights, beta=beta_schedule[epoch], cross_view_terms_only=True)
            elif model in ["MVAE", "MoPoE-VAE"]:
                train_loss += m.step([x1_sub, x2_sub, y_sub], [mask_x1, mask_x2, mask_y], modality_weights=modality_weights, beta=beta_schedule[epoch])
        # evaluate
        df_Rsq = m.quantify_cross_view_pred(mv_dataset_test.datasets[0], mv_dataset_test.datasets[1])
        df_separation = m.quantify_separation_in_latent_space(mv_helper_train.datasets, y_train.reshape(-1), mv_helper_test.datasets, y_test.reshape(-1))
        # logging R^2 for 10 first features
        logging_df = df_Rsq[df_Rsq.feature < 10]
        df_list.append(df_Rsq)
        # print training loss
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch:3d} loss {train_loss}")
        # wandb logging
        if wandb_logging:
            wandb.log({'train_loss': train_loss})
            if (epoch+1) % 10 == 0:
                wandb.log({
                    f"view_{row['view']:.0f}_feature_{row['feature']:.0f}": row.Rsq_cross for (idx, row) in logging_df.iterrows()
                })
                wandb.log({
                    "separation_shared": df_separation.iloc[1].accuracy, 
                    "AUC_shared": df_separation.iloc[1].AUC
                })
    # extract z values
    z_all = m.extract_latent_coords(mv_helper_all.datasets)
    df_z = pd.DataFrame(z_all, columns=[f"z_{i}" for i in range(z_all.shape[1])])
    return df_Rsq, df_separation, df_z
