import pandas as pd
import numpy as np
import gdown
import pyarrow.feather as feather
from .utils import concat_top_and_bottom, normalise_columns
from .generate_toy_data import generate_multimodal_data_from_GP



def get_BRCA_exprs_and_methylation(n_genes_top=None, n_genes_bottom=None, n_cpgs_top=None, n_cpgs_bottom=None, center_and_scale=True, return_M_values=True):
    path0 = "downloads/BRCA/exprs_and_meth/"
    gdown.cached_download("https://drive.google.com/uc?id=1FWAsMFvVFthgnJxbkc18hJTiGzUWb2y_", path=f"{path0}/exprs.feather", quiet=False)
    gdown.cached_download("https://drive.google.com/uc?id=1jpaGvA5l_UeJ-UsxSR4I-6xJXakEVu4p", path=f"{path0}/labels.feather", quiet=False)
    gdown.cached_download("https://drive.google.com/uc?id=1FHLPBpPF9vsgdG-zwMymfEiJE3EMt91_", path=f"{path0}/meth.feather", quiet=False)

    exprs = feather.read_feather(f"{path0}/exprs.feather").values
    meth = feather.read_feather(f"{path0}/meth.feather").values
    y = feather.read_feather(f"{path0}/labels.feather").values

    x1 = concat_top_and_bottom(exprs, n_genes_top, n_genes_bottom)
    
    x2 = concat_top_and_bottom(meth, n_cpgs_top, n_cpgs_bottom)
    
    if center_and_scale:
        x1 = normalise_columns(x1)
        if return_M_values:
            # converting to M-values
            x2 = np.log(x2 / (1.0 - x2))
            x2 = normalise_columns(x2)
    return x1, x2, y

def get_MOFA_CLL_data_IGHV(n_genes_top=None, n_genes_bottom=None, n_cpgs_top=None, n_cpgs_bottom=None, center_and_scale=True):

    exprs = pd.read_csv("downloads/CLL/exprs_IGHV.csv").values
    meth = pd.read_csv("downloads/CLL/meth_IGHV.csv").values
    labels = pd.read_csv("downloads/CLL/labels_IGHV.csv").values
    
    x1 = concat_top_and_bottom(exprs, n_genes_top, n_genes_bottom)
    
    x2 = concat_top_and_bottom(meth, n_cpgs_top, n_cpgs_bottom)
    
    if center_and_scale:
        x1 = normalise_columns(x1)
        x2 = normalise_columns(x2)
    return x1, x2, labels


def get_BRCA_exprs_ER_status(n_genes_top, n_genes_bottom, center_and_scale=True, version="4k"):
    assert version in ["4k", "10k"]
    if version == "4k":
        gdown.cached_download("https://drive.google.com/uc?id=1I6x6Orua7L-EjevUgHKzUvlsGLZSKkBE", path="downloads/BRCA/exprs_ordered_by_ER_4k.csv", quiet=False)
        df_exprs = pd.read_csv("downloads/BRCA/exprs_ordered_by_ER_4k.csv")
    elif version == "10k":
        gdown.cached_download("https://drive.google.com/uc?id=1fSaOXFU_Ga8AS0y5tTcqVRg_p6Mxft6y", path="downloads/BRCA/exprs_ordered_by_ER_10k.csv", quiet=False)
        df_exprs = pd.read_csv("downloads/BRCA/exprs_ordered_by_ER_10k.csv")

    gdown.cached_download("https://drive.google.com/uc?id=1E_nDuP31J9OXTRbgajI47m8hKat2srSv", path="downloads/BRCA/ER_status.csv", quiet=False)
    df_label = pd.read_csv("downloads/BRCA/ER_status.csv")

    exprs = df_exprs.values
    n_genes_total = exprs.shape[1]

    top = 2*n_genes_top
    bottom = n_genes_total-n_genes_bottom
    x1 = exprs[:, 0:top:2]
    x2 = np.hstack((exprs[:, 1:top:2], exprs[:, bottom:]))
    if center_and_scale:
        x1 = normalise_columns(x1)
        x2 = normalise_columns(x2)
    return x1, x2, df_label.er_status.values


def get_MOFA_RNA_ATAC(n_features_top, n_bottom_RNA, n_bottom_ATAC, scale=False):
    # exprs data
    gdown.cached_download("https://drive.google.com/uc?id=1s-HutwK0oC8PyQnIYp_s8zdrMIjFZMOP", path="downloads/MOFA_RNA_ATAC/CD8/exprs.feather", quiet=False)
    X_rna = feather.read_feather("downloads/MOFA_RNA_ATAC/CD8/exprs.feather").values

    # ATAC data
    gdown.cached_download("https://drive.google.com/uc?id=1q4Hs2CyHrwOslU5IjTs_mWIVrsYhZy86", path="downloads/MOFA_RNA_ATAC/CD8/ATAC_distal.feather", quiet=False)
    X_atac_distal = feather.read_feather("downloads/MOFA_RNA_ATAC/CD8/ATAC_distal.feather").values

    # labels 
    gdown.cached_download("https://drive.google.com/uc?id=1ER-Nr-BoojAuoEdrMY2XDCVuPWhJIWcv", path="downloads/sc_RNA_ATAC/cell_types.feather", quiet=False)
    df_labels = feather.read_feather("downloads/sc_RNA_ATAC/cell_types.feather")

    # subset to top 3000 highly variable genes
    X_rna = X_rna[:, 0:3000]
    X_rna = concat_top_and_bottom(X_rna, n_features_top, n_bottom_RNA)
    X_atac_distal = concat_top_and_bottom(X_atac_distal, n_features_top, n_bottom_ATAC)
    return X_rna, X_atac_distal, df_labels


def load_mydataset(dataset, n_genes_top=10, n_genes_bottom=10, n_cpgs_top=10, n_cpgs_bottom=1000, center_and_scale=True, **kwargs):
    if dataset == "MOFA_CLL":
        x1, x2, y = get_MOFA_CLL_data_IGHV(n_genes_top, n_genes_bottom, n_cpgs_top, n_cpgs_bottom, center_and_scale)
    elif dataset == "BRCA_ER":
        # here y = ER-status
        x1, x2, y = get_BRCA_exprs_ER_status(n_genes_top=n_genes_top, n_genes_bottom=n_genes_bottom, version="10k")
    elif dataset == "BRCA_ER_exprs_and_meth":
        x1, x2, y = get_BRCA_exprs_and_methylation(n_genes_top, n_genes_bottom, n_cpgs_top, n_cpgs_bottom, center_and_scale, return_M_values=True)
    elif dataset == "MOFA_RNA_ATAC":
        x1, x2, df_labels = get_MOFA_RNA_ATAC(n_genes_top, n_bottom_RNA=n_genes_bottom, n_bottom_ATAC=5000, scale=False)
        y = 1.0 * df_labels.celltype.isin(["naive CD8 T cells"]).values.reshape(-1, 1)
    elif dataset == "toy_GP":
        n_features = {
            "shared1": n_genes_top, 
            "shared2": n_cpgs_top, 
            "private1": n_genes_bottom, 
            "private2": n_cpgs_bottom, 
        }
        # dim_z = {"shared": 1, "private": 5}
        dim_z = kwargs.get("dim_z")
        # split by + and convert to int
        dim_z_list = [int(x) for x in dim_z.split("+")]
        x1, x2, y = generate_multimodal_data_from_GP(n_features, dim_z_list, N=1000)
    else:
        raise ValueError("Unknown dataset name")
    return x1, x2, y.reshape(-1, 1)

