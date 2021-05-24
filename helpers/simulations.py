import os
import pickle
import time
from .oblique import *

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


def simulate_maxmin_angle(max_K, samples_per_cls=5, if_save=False, save_path=None):
    loss_CE_manifold = make_lr_weight_decay(0, 0, 0.01)
    doc = {"min": numpy.zeros((max_K, max_K)), "max": numpy.zeros((max_K, max_K))}
    data_store = {}
    for K in range(1, max_K + 1):
        for M in range(1, K):
            print(f"current: K = {K}, M = {M}")
            # solver = SteepestDescent(mingradnorm=1e-5)
            solver = TrustRegions(mingradnorm=1e-5)
            manifold = Product((Oblique(M, K), Oblique(M, samples_per_cls * K), Euclidean(K)))
            Xopt, data = solve_prob(loss_CE_manifold, solver, manifold)
            key = (K, M)
            data_store[key] = Xopt
            W, H, b = Xopt
            doc["min"][K - 1, M], doc["max"][K - 1, M] = find_min_max_inner_prod(
                W)  # valid: K: 0 -> max_K - 1, M: 1 -> K - 1

    out_dict = {"doc": doc, "data_store": data_store}
    if if_save:
        if save_path is None:
            time_stamp = f"{time.time()}".replace(".", "_")
            save_path = f"parameters/general_sim_{time_stamp}.pkl"

        with open(save_path, "wb") as wf:
            pickle.dump(out_dict, wf)

    return out_dict


def fix_K_change_M(K_list, samples_per_cls=5, save_root="./parameters"):
    """
    For each K in K_list, M is from 1 to K - 1. This simulation is less clearer than "fix_M_change_M(.)".
    """
    loss_fn = make_lr_weight_decay(0, 0, 0.01)
    for K in K_list:
        print(f"current: {K} out of {K_list}")
        save_path = os.path.join(save_root, f"K_{K}.pkl")
        if os.path.exists(save_path):
            print("fetching stored data...")
            continue

        vals = numpy.zeros((K,))
        data_store = {}
        for M in range(1, K):
            print(f"current: K = {K}, M = {M}")
            # solver = SteepestDescent(mingradnorm=1e-5)
            solver = TrustRegions(mingradnorm=1e-5)
            manifold = Product((Oblique(M, K), Oblique(M, samples_per_cls * K), Euclidean(K)))
            Xopt, data = solve_prob(loss_fn, solver, manifold)

            W, H, b = Xopt
            vals[M] = find_min_max_inner_prod(W)[1]
            data_store[M] = Xopt

        with open(save_path, "wb") as wf:
            pickle.dump({"doc": vals, "data_store": data_store}, wf)


def fix_M_change_K(M_list, samples_per_cls=5, max_K_factor=4, save_root="./parameters"):
    """
    For each M in M_list, K is from M + 1 to M * max_K_factor.
     """
    loss_fn = make_lr_weight_decay(0, 0, 0.01)
    for M in M_list:
        print(f"current: {M} out of {M_list}")
        save_path = os.path.join(save_root, f"M_{M}.pkl")
        if os.path.exists(save_path):
            print("fetching stored data...")
            continue

        max_K = M * max_K_factor
        vals = numpy.zeros((max_K + 1,))
        data_store = {}
        for K in range(M + 1, max_K + 1):
            print(f"current: K = {K}, M = {M}")
            # solver = SteepestDescent(mingradnorm=1e-5)
            solver = TrustRegions(mingradnorm=1e-5)
            manifold = Product((Oblique(M, K), Oblique(M, samples_per_cls * K), Euclidean(K)))
            Xopt, data = solve_prob(loss_fn, solver, manifold)

            W, H, b = Xopt
            vals[K] = find_min_max_inner_prod(W)[1]
            data_store[K] = Xopt

        with open(save_path, "wb") as wf:
            pickle.dump({"doc": vals, "data_store": data_store}, wf)


def reload_doc(path: str):
    assert path.find(".pkl") != -1, "must be a .pkl file"
    with open(path, "rb") as rf:
        data_reload = pickle.load(rf)

    return data_reload


def plot_general_simulation(data_file):
    """
    Wrap up plots for "simulate_maxmin_angle(.)". As this is only a preliminary result, these plots are just purposed
    for a brief view.
    """
    data_reload = reload_doc(data_file)
    doc = data_reload["doc"]
    max_K = doc["min"].shape[0]

    # x axis: M, fix K
    print("Fix K, change M:")
    for K in range(1, max_K + 1):
        valid_entries_min, valid_entries_max = doc["min"][K - 1, 1: K], doc["max"][K - 1, 1: K]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
        xx = numpy.arange(1, K)
        axes[0].plot(xx, valid_entries_min)
        axes[0].set_title(f"K = {K}, min")
        axes[1].plot(xx, valid_entries_max)
        axes[1].set_title(f"K = {K}, max")
        for axis in axes:
            axis.set_xticks(xx)
        plt.show()

    print("-" * 100)
    # x axis: K, fix M
    for M in range(1, max_K):
        valid_entries_min, valid_entries_max = doc["min"][M:, M], doc["max"][M:, M]  ### beware of K's index
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
        xx = numpy.arange(M + 1, max_K + 1)
        axes[0].plot(xx, valid_entries_min)
        axes[0].set_title(f"M = {M}, min")
        axes[1].plot(xx, valid_entries_max)
        axes[1].set_title(f"M = {M}, max")
        for axis in axes:
            axis.set_xticks(xx)
        plt.show()


def plot_fix_K(K_list):
    for K in K_list:
        data_path = f"./parameters/K_{K}.pkl"
        if not os.path.exists(data_path):
            print(f"K = {K} hasn't been simulated.")
            continue
        data_reload = reload_doc(data_path)
        vals = data_reload["doc"]
        fig, axis = plt.subplots(figsize=(20, 4.8))
        xx = numpy.arange(1, K)
        axis.set_xticks(xx)
        axis.plot(xx, vals[xx])
        axis.set_title(f"K = {K}")
        axis.grid(True)
        plt.savefig(f"images/K_{K}.png")


def plot_fix_M(M_list, make_super_fig=False):
    for M in M_list:
        data_path = f"./parameters/M_{M}.pkl"
        if not os.path.exists(data_path):
            print(f"M = {M} hasn't been simulated.")
            continue
        data_reload = reload_doc(data_path)
        vals = data_reload["doc"]
        fig, axis = plt.subplots(figsize=(20, 4.8))
        max_K = vals.shape[0] - 1
        xx = numpy.arange(M + 1, max_K + 1)
        axis.set_xticks(xx)
        axis.plot(xx, vals[xx])
        axis.set_title(f"M = {M}")
        axis.grid(True)
        plt.savefig(f"images/M_{M}.png")

    if make_super_fig:
        fig, axes = plt.subplots(numpy.ceil(len(M_list) // 2).astype(numpy.int64), 2, figsize=(32, 32))
        for M, axis in zip(M_list, axes.flatten()):
            data_path = f"./parameters/M_{M}.pkl"
            if not os.path.exists(data_path):
                print(f"M = {M} hasn't been simulated.")
                continue
            data_reload = reload_doc(data_path)
            vals = data_reload["doc"]
            max_K = vals.shape[0] - 1
            xx = numpy.arange(M + 1, max_K + 1)
            axis.set_xticks(xx)
            axis.plot(xx, vals[xx])
            axis.set_title(f"M = {M}")
            axis.grid(True)

        time_stamp = f"{time.time()}".replace(".", "_")
        path = f"images/super_fig_{time_stamp}.png"
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':
    # simulate_mcr(num_cls=2)

    # simulate_etf_K_gt_M(K=5)

    pass
