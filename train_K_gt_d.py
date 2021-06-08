import argparse
import torch
import torch.nn as nn
import geoopt
import os
from run_cifar100.datasets import select_k_cls
from run_cifar100.models import ResNetAdapt
from run_cifar100.utils import train_epoch, save_model, create_gif


def read_in_args():
    parser = argparse.ArgumentParser(description="Train a model with K greater than d.")
    parser.add_argument("--epoch", type=int, help="number of epochs", required=True)
    parser.add_argument("--batch-size", type=int, help="batch size to use", default=128)
    parser.add_argument("--classes", type=int, help="number of classes", dest="K", required=True)
    parser.add_argument("--feature-dim", type=int, help="feature dimension", dest="M", required=True)
    parser.add_argument("--scheduler", action="store_true", help="whether to use StepLR Scheduler")
    parser.add_argument("--feature-man", action="store_true",
                        help="whether constrain last layer's feature on sphere", dest="feature_man")
    parser.add_argument("--weight-man", action="store_true",
                        help="whether constrain linear classifier's weight on sphere", dest="W_man")
    parser.add_argument("--lr", type=float, default=2e-2)

    args = parser.parse_args()

    return args.__dict__


def train_cli(**kwargs):
    M, K = kwargs["M"], kwargs["K"]
    epochs = kwargs["epoch"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["lr"]

    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader, test_loader = select_k_cls(num_cls=K, batch_size=batch_size)
    model = ResNetAdapt(M, K, oblique_feature=kwargs["feature_man"], oblique_weight=kwargs["W_man"]).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = geoopt.optim.RiemannianAdam([param for param in model.parameters() if param.requires_grad], lr=lr)
    scheduler = None
    if kwargs["scheduler"]:
        step_size = 10
        gamma = 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    train_args = {"epoch": epochs, "validate_batch": [], "validate_epoch": [i for i in range(epochs) if i % 5 == 0],
                  "feature_decay": 0.01, "weight_decay": 0.01, "criterion": criterion}

    out_dict = {}
    tr_loss, tr_acc, val_loss, val_acc = train_epoch(model, train_loader, test_loader, opt, device,
                                 train_args, scheduler=scheduler, out_dict=out_dict)

    weights = out_dict["weights"]
    log_dict = {"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc}

    cifar100_dir = "./cifar100_params"
    if not os.path.exists(cifar100_dir):
        os.mkdir(cifar100_dir)

    base = (f"{cifar100_dir}/epoch_{epochs}_K_{K}_{kwargs['W_man']}_M_{M}_{kwargs['feature_man']}_"
            + "{data_type}.{suffix}").format

    save_model(model, base(data_type="model", suffix="pt"), log_dict, base(data_type="curve", suffix="pkl"))

    create_gif(weights, if_save=True, path=base(data_type="dymanics", suffix="gif"))


if __name__ == '__main__':
    args = read_in_args()
    train_cli(**args)
