import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm.notebook import tqdm
from typing import List


def train(model, train_loader, test_loader, optimizer, device, epoch, train_args):
    model.train()
    loss = 0
    validate_batches = train_args["validate_batch"]
    criterion = train_args["criterion"]
    local_tr_loss, local_val_loss, local_tr_acc, local_val_acc = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    for i, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X = X.to(device)
        y = y.to(device)
        X_out, f_norm = model(X)
        # print(X_out.shape) ###
        # print(y)  ###
        loss = criterion(X_out, y)

        if not model.oblique_weight:
            loss += torch.norm(model.last_W) ** 2 * 0.5 * train_args["weight_decay"]
        if not model.oblique_feature:
            loss += f_norm ** 2 * 0.5 * train_args["feature_decay"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, indices = torch.max(X_out, dim=-1)
            acc = ((indices == y).sum() / len(y)).item()

        if i in validate_batches or i == len(train_loader) - 1:
            key = f"{epoch}.{i}"
            local_tr_loss[key] = loss.item()
            local_tr_acc[key] = acc
            local_val_loss[key], local_val_acc[key] = eval(model, test_loader, device, train_args)
            print(f"batch {i + 1}/{len(train_loader)}")
            print(f"training loss: {local_tr_loss[key]}, val loss: {local_val_loss[key]}")
            print(f"training acc: {local_tr_acc[key]}, val acc: {local_val_acc[key]}")

    return local_tr_loss, local_tr_acc, local_val_loss, local_val_acc


@torch.no_grad()
def eval(model, test_loader, device, train_args):
    model.eval()
    loss_avg = 0
    num_samples = 0
    num_correct_preds = 0

    for X, y in tqdm(test_loader, total=len(test_loader)):
        X = X.to(device)
        y = y.to(device)
        X_out, f_norm = model(X)
        loss = criterion(X_out, y)

        if not model.oblique_weight:
            loss += torch.norm(model.last_W) ** 2 * 0.5 * train_args["weight_decay"]
        if not model.oblique_feature:
            loss += f_norm ** 2 * 0.5 * train_args["feature_decay"]

        loss_avg += loss.item() * len(y)
        num_samples += len(y)
        _, indices = torch.max(X_out, dim=-1)
        num_correct_preds += (indices == y).sum()

    return loss_avg / num_samples, num_correct_preds / num_samples


def train_epoch(model, train_loader, test_loader, optimizer, device, train_args, scheduler=None, out_dict=None):
    """
    out_dict: Anything wanting to store
    """
    tr_loss, val_loss, tr_acc, val_acc = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    epochs = train_args["epoch"]
    validate_epochs = train_args["validate_epoch"]

    for epoch in tqdm(range(epochs)):
        print(f"epoch {epoch + 1}/{epochs}")
        local_tr_loss, local_tr_acc, local_val_loss, local_val_acc = train(model, train_loader, test_loader,
                                                                           optimizer, device, epoch, train_args)
        tr_loss.update(local_tr_loss)
        tr_acc.update(local_tr_acc)
        val_loss.update(local_val_loss)
        val_acc.update(local_val_acc)

        # (Optional) Save the halfway-through model here
        ################################
        if epoch in validate_epochs:
            model.plot_last_W()
            if out_dict is not None:
                weights_list = out_dict.get("weights", [])
                weights_list.append(model.last_W.detach().cpu().numpy())
                out_dict["weights"] = weights_list
        ################################

        if scheduler is not None:
            scheduler.step()

    return tr_loss, tr_acc, val_loss, val_acc


def save_model(model, model_path, log_dict, log_path):
    """
    log_dict: {"train_loss": tr_loss, ...}
    """
    torch.save(model.state_dict(), model_path)
    with open(log_path, "wb") as wf:
        pickle.dump(log_dict, wf)


def reload_model(model_to_reload, model_path, src="cuda", tgt="cuda"):
    device = torch.device(tgt)
    if src == tgt:
        model_to_reload.load_state_dict(torch.load(path))
        model_to_reload.to(device).eval()
        return model_to_reload

    else:
        if tgt == "cuda":
            tgt = "cuda:0"
        model_to_reload.load_state_dict(torch.load(path, map_location=tgt))
        model_to_reload.to(device).eval()

    return model_to_reload


def plot_curve(log_dict, figsize=(18, 9.6)):
    """
    log_dict: keys: "train_loss", "val_loss", "train_acc", "val_acc"
    """
    def extract_epoch_keys(keys):
        epoch_keys = []
        for i in range(len(keys) - 1):
            cur_key = keys[i]
            next_key = keys[i + 1]

            if cur_key.split(".")[0] != next_key.split(".")[0]:
                epoch_keys.append(cur_key)
        epoch_keys.append(keys[-1])
        epoch_keys_labels = [f"{int(epoch_key.split('.')[0]) + 1}" for epoch_key in epoch_keys]

        return epoch_keys, epoch_keys_labels

    tr_loss, val_loss = log_dict["train_loss"], log_dict["val_loss"]
    tr_acc, val_acc = log_dict["train_acc"], log_dict["val_acc"]

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    labels = ["train", "eval"]
    keys = list(tr_loss.keys())

    for i, loss_dict in enumerate([tr_loss, val_loss]):
        axes[0].plot(keys, list(loss_dict.values()), label=labels[i])

    for i, loss_dict in enumerate([tr_acc, val_acc]):
        axes[1].plot(keys, list(loss_dict.values()), label=labels[i])

    epoch_keys, epoch_labels = extract_epoch_keys(keys)

    ylabels = ["loss", "acc"]
    for i, axis in enumerate(axes.ravel()):
        axis.set_xticks(epoch_keys)
        axis.set_xticklabels(epoch_labels)
        axis.grid(True)
        axis.legend()
        axis.set_xlabel("epochs")
        axis.set_ylabel(ylabels[i])

    plt.show()


def create_gif(weight_path: List[np.array], if_save=False, path=None):
    """
    Only supports 2D features.
    """
    # make sure we have constant frame size
    fig, axis = plt.subplots()
    first_weight = weight_path[0]
    axis.scatter(first_weight[:, 0], first_weight[:, 1])
    max_lim = max(list(map(abs, list(axis.get_xlim()) + list(axis.get_ylim()))))
    axis.set_xlim(-max_lim, max_lim)
    axis.set_ylim(-max_lim, max_lim)

    axis.grid(True)
    scat = axis.scatter([], [])

    def update_plot(i):
        scat.set_offsets(data[i])
        axis.set_title(f"epoch {i + 1}")
        return scat

    ani = animation.FuncAnimation(fig, update_plot, frames=epochs)
    plt.show()

    if if_save:
        assert path is not None, "please specify a save path if you want to save the GIF"
        ani.save(path, writer="pillow", fps=2)  # specific writer for gif


def plot_feature(data_loader, model, device):
    for X, y in data_loader:
        break

    X = X.to(device)
    y = y.to(device)
    features = model.get_feature(X).detach().cpu().numpy()
    print(features.shape)
    fig, axis = plt.subplots()
    y_np = y.detach().cpu().numpy()
    for cls in range(len(data_loader.dataset.cls_list)):
        mask = (y_np == cls)
        features_selected = features[mask, ...]
        label_idx = train_loader.dataset.cls_list[cls]
        label = train_loader.dataset.classes[label_idx]
        axis.scatter(features_selected[:, 0], features_selected[:, 1], label=label)
    axis.set_aspect("equal")
    axis.grid(True)
    axis.legend()
    plt.show()
