import numpy
import autograd.numpy as np
import matplotlib.pyplot as plt
import pymanopt
import re

from pymanopt.manifolds import Sphere, Oblique, Product, Euclidean
from pymanopt.solvers import TrustRegions, SteepestDescent
from pymanopt import Problem
from contextlib import redirect_stdout


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
    Stable multi-class CE loss. No loops.

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
        Primary paper: objective with weight, feature and bias decay.

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
        assert H.shape[-1] % num_cls == 0, "number of samples must be a multiple of number of classes"
        assert W.shape[0] >= W.shape[-1], "number of features must be no less than number of training samples"
        N = H.shape[-1] // num_cls  # number of training samples per class
        cost = 0.5 * (lam_W * np.linalg.norm(W) ** 2 + lam_H * np.linalg.norm(H) ** 2 + lam_b * np.linalg.norm(b) ** 2)
        cls_labels = np.kron(np.arange(num_cls), np.ones((N,))).astype(np.int64)
        softmax_features = W.T @ H + b.reshape((-1, 1))

        # stable softmax
        cost += softmax_loss(softmax_features, cls_labels)

        return cost

    return lr_weight_decay


def make_mcr2_loss(num_cls, eps=1e-2):
    def mcr2_loss(Z):
        """
        $MCR^2$ loss, assuming all classes have equal number of training samples.

        Parameters
        ----------
        Z: numpy.ndarray
            In the form of [z1, ..., zn], and samples are properly sorted: [Z1, ..., ZC] where C is number of classes.

        Returns
        -------
        float
        """
        N, M = Z.shape
        assert M % num_cls == 0, "number of training samples should be a multiple of number of classes"
        M_per_cls = M // num_cls
        alpha = N / (M * eps ** 2)
        alpha_j = N / (M_per_cls * eps ** 2)
        gamma_j = M_per_cls / M
        outer = Z @ Z.T
        cost = 0.5 * np.log(np.linalg.det(np.eye(N) + alpha * outer))

        for i in range(0, M, M_per_cls):
            Z_i = Z[:, i : i + M_per_cls]
            cost -= 0.5 * gamma_j * np.log(np.linalg.det(np.eye(N) + alpha_j * Z_i @ Z_i.T))

        return -cost

    return mcr2_loss


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


def convert_stdout_to_data(filename, mode="RGD"):
    """
    Pymanopt directly prints cost to console, but it's better to visualize the optimization process. This function
    converts the log file to a numpy array. Only works for SteepestDescent solver.

    Parameters
    ----------
    filename: str
        Log file.
    mode: str
        "RGD" or "RTR"

    Returns
    -------
    numpt.ndarray
    """
    data = []
    if mode == "RGD":
        with open(filename, "r") as rf:
            for i in range(3):
                rf.readline()

            line = rf.readline().strip()
            while len(line) > 0:
                raw = line.split()[1:]
                # print(raw)
                try:
                    data.append(list(map(float, raw)))
                except ValueError:
                    pass
                line = rf.readline().strip()
    elif mode == "RTR":
        with open(filename, "r") as rf:
            for i in range(4):
                rf.readline()

            line = rf.readline().strip()
            while len(line) > 0:
                try:
                    fval_raw = re.findall(r"f: [-+0-9e.]+", line)[0]
                    # print(fval_raw)
                    grad_raw = re.findall(r"\|grad\|: [-+0-9e.]+", line)[0]
                    # print(grad_raw)
                    data.append([float(fval_raw[len("f: "):]), float(grad_raw[len("|grad|: "):])])
                except IndexError:
                    pass
                line = rf.readline().strip()

    data = numpy.array(data)

    return data


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot

    # MCR^2, interactive 3D plot
    num_cls = 2  # 3 or 2
    sample_per_cls = 10
    N = sample_per_cls * num_cls
    M = 3
    solver = TrustRegions()
    # solver = SteepestDescent()
    manifold = Oblique(M, N)
    prob = Problem(manifold=manifold, cost=make_mcr2_loss(num_cls))
    with open("loss.txt", "w") as wf:
        with redirect_stdout(wf):
            Xopt = solver.solve(prob)

    data = convert_stdout_to_data("loss.txt", "RTR")

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(data[:, 0])
    axes[0].set_title("fval")
    axes[1].plot(data[:, 1])
    axes[1].set_title("norm of grad")

    # Xopt *= numpy.sign(Xopt[0, :])  # get rid of sign flip, but better not to do it to see obvious orthogonality
    fig = plt.figure(figsize=(8, 8))
    axis = fig.gca(projection="3d")
    colors = ["r", "b", "g"]
    for cls in range(num_cls):
        label = cls + 1
        start = cls * sample_per_cls
        end = (cls + 1) * sample_per_cls
        axis.scatter(*[Xopt[i, start: end] for i in range(Xopt.shape[0])], color=colors[cls], label=f"cls {label}")
    axis.grid(True)
    axis.set_xlim([-1, 1])
    axis.set_ylim([-1, 1])
    axis.set_zlim([-1, 1])
    axis.legend()
    plt.show()  # You should be able to see subspaces are orthogonal to each other
