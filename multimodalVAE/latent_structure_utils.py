import torch

def get_latent_structure(key):
    """
    Create private-shared mask for latent structure.
    """
    sparsity_mask_options = {
        "2+2+2": torch.Tensor([
            [1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0]
        ]),
        "3+3+3": torch.Tensor([
            [1, 1, 1, 1, 1, 1, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0]
        ]),
        "4+4+4": torch.Tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        ]),
        "5+5+5": torch.Tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ]),
        "10+10+10": torch.Tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    }
    return sparsity_mask_options[key]