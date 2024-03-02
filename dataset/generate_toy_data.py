import numpy as np
import torch

def RBF(x, y, lengthscale, variance, jitter=None):
    N = x.shape[0]
    x = x / lengthscale
    y = y / lengthscale
    s_x = np.sum(x ** 2, axis=1).reshape([-1, 1])
    s_y = np.sum(y ** 2, axis=1).reshape([1, -1])
    K = variance * np.exp(- 0.5 * (s_x + s_y - 2 * np.dot(x, y.T)))
    if jitter is not None:
        K += jitter * np.eye(N)
    return K


def generate_from_MVN(K, n_features):
    N = K.shape[0]
    # shape [N, N]
    L = np.linalg.cholesky(K)
    # shape [N, P]
    eps = np.random.randn(N, n_features)
    # shape [N, P]
    return np.dot(L, eps)

def generate_from_GP(z, n_features):
    # ls = np.random.uniform(0.5, 2.0, size=z.shape[1])
    ls = 1.0
    K = RBF(z, z, ls, 1.0, jitter=1e-6)
    y = generate_from_MVN(K, n_features)
    return y

def generate_multimodal_data_from_GP(n_features, dim_z_list, N):

    # make sure n_features contains the keys "shared1", "shared2", "private1", "private2"
    assert "shared1" in n_features, "n_features must contain 'shared1'"
    assert "shared2" in n_features, "n_features must contain 'shared2'"
    assert "private1" in n_features, "n_features must contain 'private1'"
    assert "private2" in n_features, "n_features must contain 'private2'"

    def _add_noise_and_normalise(x, noise_sd):
        # normalise
        x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
        # add noise
        x += noise_sd * np.random.randn(*x.shape)
        return x

    # split "A+B+C" into ["A", "B", "C"] where B denotes the dimension of the z_shared
    z_shared = np.random.uniform(-2.0, 2.0, size=[N, dim_z_list[1]])
    x1_shared = generate_from_GP(z_shared, n_features["shared1"])
    x2_shared = generate_from_GP(z_shared, n_features["shared2"])
    # private1
    z_pr1 = np.random.uniform(-2.0, 2.0, size=[N, dim_z_list[0]])
    x1_private_features = generate_from_GP(z_pr1, n_features["private1"])
    # private2
    z_pr2 = np.random.uniform(-2.0, 2.0, size=[N, dim_z_list[2]])
    x2_private_features = generate_from_GP(z_pr2, n_features["private2"])
    # x1 and x2
    x1 = np.hstack((x1_shared, x1_private_features))
    x2 = np.hstack((x2_shared, x2_private_features))
    y = 1.0 * (z_shared[:, 0] > 0.0)
    # add noise and normalise
    x1 = _add_noise_and_normalise(x1, noise_sd=0.15)
    x2 = _add_noise_and_normalise(x2, noise_sd=0.15)
    return x1, x2, y

