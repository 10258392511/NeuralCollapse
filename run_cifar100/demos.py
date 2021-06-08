import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


def demo_manifold_tensor(shape=(3, 2)):
    """
    f(W) = sum(W @ a)
    Demonstrate manifold.projx(.) (manifold_tensor.proj_()), manifold.proju(.), manifold.retr(.).
    """
    W_init = geoopt.ManifoldTensor(torch.randn(shape), manifold=geoopt.Sphere())  # oblique
    manifold = W_init.manifold
    W = geoopt.ManifoldTensor(W_init.proj_(), manifold=geoopt.Sphere(), requires_grad=True)
    a = torch.randn(2)
    b = (W @ a).sum()
    b.backward()
    print(f"W: {W}\nnorm: {W.norm(dim=1)}")

    with torch.no_grad():
        stepsize = 0.1
        rgrad = manifold.egrad2rgrad(W, W.grad)
        W_next = manifold.retr(W, stepsize * rgrad)
        print(f"W_next: {W_next}\nnorm: {W_next.norm(dim=1)}")

        diag_ele = torch.diag(W @ rgrad.T)
        assert np.allclose(diag_ele.numpy(), 0, atol=1e-6), f"rgrad is not orthogonal to W: {diag_ele}"


class LastLayer(nn.Module):
    def __init__(self, M, K):
        super().__init__()
        W_init = geoopt.ManifoldTensor(torch.randn((K, M)), manifold=geoopt.Sphere())
        manifold = W_init.manifold
        self.W = geoopt.ManifoldParameter(manifold.projx(W_init), manifold=geoopt.Sphere())  # (K, M), row-wise
        self.b = geoopt.ManifoldParameter(torch.ones((K,)))

    def forward(self, x):
        # x: (B, M), W: (K, M)
        return x @ self.W.T + self.b


def demo_manifold_param(M=2, K=4):
    """
    Demonstrates initializing a ManifoldParameter, RiemannianSGD (optimizer).

    Parameters
    ----------
    M: int
        Feature dimension.
    K: int
        Number of classes.
    """
    M, K = 2, 4
    model = LastLayer(M, K)
    with torch.no_grad():
        print(f"{model.W}\n{model.W.norm(dim=1)}")  # should all be unit-norm, row-wise

    opt = geoopt.optim.RiemannianSGD([param for param in model.parameters() if param.requires_grad], lr=0.01,
                                     momentum=0.95, nesterov=True)

    B = 3
    X = torch.randn((B, M))  # (B, M), W: (K, M)
    X_out = model(X)
    y_out = X_out.sum()  # L = sum(x @ W.T + b)
    opt.zero_grad()
    y_out.backward()
    opt.step()

    with torch.no_grad():
        print(f"{model.W}, {model.W.norm(dim=1)}")  # should be different from init but still unit-norm
