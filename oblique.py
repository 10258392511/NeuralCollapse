import numpy
import autograd.numpy as np
import matplotlib.pyplot as plt
import pymanopt
from pymanopt.manifolds import Sphere, Oblique, Product, Euclidean
from pymanopt.solvers import TrustRegions, SteepestDescent
from pymanopt import Problem


def lr_cost(U, alpha=1):
    """
    "Neural Collapse with CE Loss" paper: objective of problem 1. Slow version.

    Parameters
    ----------
    U: numpy.ndarray
        Current location
    alpha: float
        Temperature of softmax(.)

    Returns
    -------
    float
    """
    M, N = U.shape
    assert M >= N, "number of features should be no less than number of samples"
    cost = 0
    for i in range(N):
        log_term = 0
        for j in range(N):
            if j != i:
                log_term += np.exp(alpha * U[:, i] @ U[:, j] - 1)

        cost += np.log(1 + log_term)

    return cost


def lr_cost_weights(X):
    """
    "Neural Collapse with CE Loss" paper: objective of problem 2. Slow version.

    Parameters
    ----------
    X: numpy.ndarray
        Current location, in the form of [U, V].

    Returns
    -------
    float
    """
    _, N = X.shape
    assert N % 2 == 0, "number of columns must be even"
    num_cls = N // 2
    U, V = X[:, :num_cls], X[:, num_cls:]
    cost = 0

    for i in range(num_cls):
        log_term = 0
        for j in range(num_cls):
            if i != j:
                log_term += np.exp((V[:, j] - V[:, i]) @ U[:, i])

        cost += log_term

    return cost


def softmax_loss(softmax_features, labels):
    """
    Stable multi-class CE loss.

    Parameters
    ----------
    softmax_features: numpy.ndarray
        Each column is a feature.
    labels: numpy.ndarray
        Of shape (number of samples,)

    Returns
    -------
    float
    """
    max_col = softmax_features.max(axis=0, keepdims=True)
    # print(max_col.shape)
    features = softmax_features - max_col
    features = np.exp(features)
    features_sum_col = features.sum(axis=0, keepdims=True)
    cost = np.log(features_sum_col / features[labels, np.arange(features.shape[-1])]).sum() / softmax_features.shape[-1]

    return cost


def make_lr_weight_decay(lam_W=0.3, lam_H=0.3, lam_b=0.3):
    def lr_weight_decay(X):
        """
        Primary paper: objective with weight, feature and bias decay. No loops.

        Parameters
        ----------
        X: list
            Current location, in the form of [W, H, b], where W, H is oblique manifold; b has no constraints.
        num_cls: int
            Number of classes
        lam_W, lam_H, lam_b: float
            Regularization parameters.

        Returns
        -------
        float
        """
        # num_cls =
        # W, H, b = X[:, :num_cls], X[:, num_cls:-1], X[:, -1]
        W, H, b = X
        num_cls = b.shape[0]
        assert H.shape[-1] % num_cls == 0, "number of samples must be multiple of number of classes"
        assert W.shape[0] >= W.shape[-1], "number of features must be no less than number of training samples"
        N = H.shape[-1] // num_cls  # number of training samples per class
        cost = 0.5 * (lam_W * np.linalg.norm(W) ** 2 + lam_H * np.linalg.norm(H) ** 2 + lam_b * np.linalg.norm(b) ** 2)
        cls_labels = np.kron(np.arange(num_cls), np.ones((N,))).astype(np.int64)
        softmax_features = W.T @ H + b.reshape((-1, 1))

        # stable softmax
        cost += softmax_loss(softmax_features, cls_labels)

        return cost

    return lr_weight_decay


def check_etf(X, verbose=False, atol=1e-8):
    """
    Check whether "X" is a simplex ETF, where X = [x1, ... xn]. Raise AssertionError if any test fails; print "ETF Tests
    passed!" if successful.

    Parameters
    ----------
    X: numpy.ndarray
        Input array.
    verbose: bool
        If True, prints out norms of each column of X and pair-wise inner products.
    atol: float
        Relative tolerence, used in numpy.allclose(.).

    Returns
    -------
    None
    """
    _, N = X.shape
    # check norms
    norms = numpy.linalg.norm(X, axis=0)
    if verbose:
        print(norms)
    assert numpy.allclose((numpy.ones((N, )) * norms[0]), norms), "not equi-norm"

    # check angles
    inner_products = X.T @ X
    if verbose:
        print(inner_products)

    ground_truth = -numpy.ones((N, N)) / (N - 1)
    ground_truth[numpy.arange(N), numpy.arange(N)] = 1
    assert numpy.allclose(inner_products, ground_truth, atol=atol), "not equi-angular"

    print("ETF Tests passed!")


def check_duality(W, H, verbose=False, atol=1e-4):
    samples_per_cls = H.shape[1] // W.shape[1]
    H_from_W = numpy.kron(W, numpy.ones((samples_per_cls,)))
    check_etf(W, verbose=verbose, atol=atol)
    assert numpy.allclose(H_from_W, H, atol=atol), "duality doesn't hold"

    print("All tests passed!")
