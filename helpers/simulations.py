from oblique import *

DEFAULT_LOSS_TXT_PATH = "../log_files/loss.txt"


def simulate_mcr(num_cls=3, sample_per_cls=10):
    assert num_cls in (2, 3), "number of classes should be 2 or 3"

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot

    N = sample_per_cls * num_cls
    M = 3
    solver = TrustRegions()
    # solver = SteepestDescent()
    manifold = Oblique(M, N)
    loss_fn = make_mcr2_loss(num_cls)
    Xopt, data = solve_prob(loss_fn, solver, manifold, temp_log_file=DEFAULT_LOSS_TXT_PATH)

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


def simulate_etf_K_gt_M(K=6, samples_per_cls=5):
    """
    Simulation for K >> M cases for 3D features.

    Parameters
    ----------
    K: int
        Number of classes.
    samples_per_cls: int
        Number of samples per class.

    Returns
    -------
    None
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot

    M = 3
    solver = SteepestDescent()
    manifold = Product((Oblique(M, K), Oblique(M, samples_per_cls * K), Euclidean(K)))
    loss_fn = make_lr_weight_decay(0, 0, 0.01)
    Xopt, data = solve_prob(loss_fn, solver, manifold, temp_log_file=DEFAULT_LOSS_TXT_PATH)

    W, H, b = Xopt
    fig = plt.figure(figsize=(6, 6))
    axis = fig.gca(projection="3d")
    axis.grid(True)
    axis.scatter(W[0, :], W[1, :], W[2, :])
    axis.set_xlim([-1, 1])
    axis.set_ylim([-1, 1])
    axis.set_zlim([-1, 1])
    plt.show()


if __name__ == '__main__':
    # simulate_mcr(num_cls=2)

    # simulate_etf_K_gt_M(K=5)

    pass
