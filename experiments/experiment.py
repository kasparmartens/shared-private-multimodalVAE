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
from dataset.load_data import load_mydataset

from experiments.core import train_and_eval

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = cfg.model.name
    assert model in ["MMVAE", "MMVAE++", "MVAE", "MoPoE-VAE"]
    latent_structure = cfg.model.latent_structure
    use_labels = cfg.model.supervised
    n_epochs = cfg.trainer.n_epochs
    lr = cfg.trainer.lr

    # convert cfg to dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(project="multimodalVAE", config=cfg_dict, tags=[cfg.dataset.name, latent_structure, "final_final"], reinit=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device {device}")

    x1, x2, y = load_mydataset(
        cfg.dataset.name, 
        n_genes_top=cfg.dataset.n_genes_top, 
        n_genes_bottom=cfg.dataset.n_genes_bottom,
        # if n_cpgs_top is not specified, it will be set to n_genes_top
        n_cpgs_top=cfg.dataset.get("n_cpgs_top", cfg.dataset.n_genes_top),
        # if n_cpgs_bottom is not specified, it will be set to n_genes_bottom
        n_cpgs_bottom=cfg.dataset.get("n_cpgs_bottom", cfg.dataset.n_genes_bottom),
        # the last arg dim_z is only used when generating synthetic data
        dim_z=cfg.dataset.get("dim_z", None)
    )

    df_Rsq, df_separation, df_z = train_and_eval(
        x1=x1, x2=x2, y=y, 
        model=model, 
        latent_structure=latent_structure, 
        supervised=use_labels, 
        modality_weights=[1.0, 1.0, 100.0], 
        device=device, 
        n_epochs=n_epochs, 
        lr=lr
    )

    wandb.log({
        "df_Rsq": df_Rsq, 
        "df_separation": df_separation 
    })

    wandb.finish()


if __name__ == "__main__":
    main()
