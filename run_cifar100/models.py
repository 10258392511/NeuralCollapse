import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import geoopt

from torchvision.models import resnet18


class ResNetAdapt(nn.Module):
    """
    X = resnet18(X), ending at avgpool(.)
    X = fc(X), 512 -> M, supporting Oblique X
    X = last_layer(X) M -> K, supporting Oblique W
    """

    def __init__(self, feature_dim, num_classes,
                 oblique_feature=False, oblique_weight=False):
        super().__init__()
        self.resnet = resnet18()
        self.feature_dim = feature_dim
        self.resnet.fc = nn.Linear(self.resnet.fc.weight.shape[1], feature_dim,
                                   bias=True)
        self.num_classes = num_classes
        self.oblique_feature = oblique_feature
        self.oblique_weight = oblique_weight

        if oblique_weight:
            W = geoopt.ManifoldTensor(torch.randn((self.num_classes, self.feature_dim)),
                                      manifold=geoopt.Sphere())  # (K, M), row-wise
            manifold = W.manifold
            self.last_W = geoopt.ManifoldParameter(manifold.projx(W), manifold=geoopt.Sphere())
        else:
            W = torch.empty((self.num_classes, self.feature_dim))
            nn.init.kaiming_normal_(W)
            self.last_W = geoopt.ManifoldParameter(W)

        b = torch.empty((num_classes,))
        nn.init.uniform_(b)
        self.last_b = geoopt.ManifoldParameter(b)  # (K,)

    def forward(self, x):
        # x: (B, M), W: (K, M)
        x = self.resnet(x)
        f_norm = x.norm()
        if self.oblique_feature:
            norm = x.norm(dim=-1, keepdim=True)
            x = x / norm

            with torch.no_grad():
                assert np.allclose(x.detach().cpu().norm(dim=-1), 1)

        # print(f"inside model: W: {self.last_W.shape}, x: {x.shape}")
        x = x @ self.last_W.T + self.last_b

        return x, f_norm

    @torch.no_grad()
    def get_feature(self, x):
        x = self.resnet(x)
        if self.oblique_feature:
            norm = x.norm(dim=-1, keepdim=True)
            x = x / norm

        return x

    @torch.no_grad()
    def plot_last_W(self):
        assert self.feature_dim in [2, 3], "only can plot 2D or 3D weights"

        if self.feature_dim == 2:
            fig, axis = plt.subplots()
            W = self.last_W.detach().cpu().numpy()
            axis.scatter(W[:, 0], W[:, 1])
            axis.set_aspect("equal")
            axis.grid(True)
            axis.scatter([0], [0], marker="*")
            plt.show()
