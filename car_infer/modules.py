import torch
import torch as th
import torch.nn as nn
from Net import functional
import numpy as np
from Net import geoopt
from Net.geoopt.manifolds import *
from Net.geoopt import linalg
import torch.nn.functional as F

dtype = th.float32
device = th.device("cuda")


class BiMap(nn.Module):
    def __init__(self, h, ni, no, m):
        super(BiMap, self).__init__()

        self.increase_dim = None

        self._h = h
        self._m = m

        self._ni, self._no = ni, no

        # self._W = torch.nn.Parameter(th.empty(self._m, self._ni, self._no, dtype=dtype, device=device))
        self._W = functional.StiefelParameter(
            th.empty(self._m, self._ni, self._no, dtype=dtype)
        )
        # self._W = geoopt.ManifoldParameter(th.empty(self._m, self._ni, self._no), manifold=geoopt.CanonicalStiefel())
        self._init_bimap_parameter()
        self.W = torch.zeros_like(self._W, requires_grad=True)

        self.LogEig = LogEig()
        self.QRDecomposition = QRDecomposition()

        # self.E = None
        # self.G = None

    def _init_bimap_parameter(self):
        for i in range(self._m):
            v = th.empty(
                self._ni, self._ni, dtype=self._W.dtype, device=self._W.device
            ).uniform_(0.0, 1.0)
            vv = th.svd(v.matmul(v.t()))[0][:, : self._no]
            self._W.data[i] = vv

    def _bimap_multiplication(self, X):
        batch_size, channel_size, dim_in, _ = X.size()
        X = X.view(batch_size * channel_size, 1, dim_in, _)

        Q, R = self.QRDecomposition(self._W)
        D = th.diag_embed(th.sign(th.diagonal(R, dim1=-2, dim2=-1)))
        self.W = Q @ D
        self.W = self.W.unsqueeze(0)

        X = X.view(batch_size, channel_size, dim_in, -1)

        P = th.matmul(
            th.matmul(self.W.transpose(-1, -2), X.unsqueeze(2).to(th.float32)), self.W
        )
        # print(self.W.transpose(-1, -2).shape)
        # print(X.unsqueeze(2).shape)
        # print(self.W.shape)
        # P = self.W.transpose(-1, -2) @ X.unsqueeze(2).to(th.float32) @ self.W
        P = P.view(batch_size, channel_size * self._m, self._no, self._no)

        if self._ni > self._no:
            # E = self.euclidean_dis(X)
            # G = self.geodesic_dis(X)
            # L = self.L(X)
            L = 0
            return P, L
        else:
            return P

    def forward(self, X):
        return self._bimap_multiplication(X)

    def geodesic_dis(self, spd):
        batch_size, channels_in, dim, dim = spd.size()
        spd_inv = geoopt.linalg.sym_inv_sqrtm1(spd).repeat_interleave(
            channels_in, dim=1
        )
        spd = spd.repeat(1, channels_in, 1, 1)
        diff = self.LogEig(spd_inv @ spd @ spd_inv)
        diff = diff.view(batch_size, channels_in, channels_in, dim, dim)
        dis = torch.norm(diff, p="fro", dim=[-1, -2], keepdim=True).squeeze()
        return dis

    def euclidean_dis(self, spd):
        log_X = self.LogEig(spd).unsqueeze(1).to(torch.float32)
        diff = log_X.unsqueeze(3) - log_X.unsqueeze(2)
        # print("diff", diff.size())
        w = self.W.unsqueeze(2).unsqueeze(2)
        # print("w", w.size())
        diff = w.transpose(-1, -2) @ diff @ w
        dis = torch.mean(
            torch.norm(diff + 1e-12, p="fro", dim=[-1, -2], keepdim=True)
            .squeeze(-1)
            .squeeze(-1),
            dim=1,
        )
        return dis

    # def L(self, spd):
    # log_X = self.LogEig(spd).unsqueeze(1).to(torch.float32)
    # diff = log_X.unsqueeze(3) - log_X.unsqueeze(2)
    # print("diff", diff.size())
    # w = self.W.unsqueeze(2).unsqueeze(2)
    # print("w", w.size())
    # diff = diff @ w @ w.transpose(-1, -2) @ diff.transpose(-1, -2)
    # print("diff____", diff.size())
    # L = self.geodesic_dis(spd).unsqueeze(1).unsqueeze(-1).unsqueeze(-1) * diff
    # L = torch.sum(L, dim=2)
    # L = torch.sum(L, dim=2)
    # print("L", L.size())
    # return L


class QRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        q, r = torch.linalg.qr(input)
        ctx.save_for_backward(q, r)
        return q, r

    @staticmethod
    def backward(ctx, grad_q, grad_r):
        q, r = ctx.saved_tensors
        grad_input = torch.matmul(q, grad_r) + torch.matmul(
            grad_q, torch.transpose(r, 1, 2)
        )
        return grad_input


class BatchNormSPD(nn.Module):
    def __init__(self, momentum, n):
        super(__class__, self).__init__()

        self.momentum = momentum

        self.running_mean = geoopt.ManifoldParameter(
            th.eye(n, dtype=th.double),
            manifold=geoopt.SymmetricPositiveDefinite(),
            requires_grad=False,
        )
        self.weight = geoopt.ManifoldParameter(
            th.eye(n, dtype=th.double),
            manifold=geoopt.SymmetricPositiveDefinite(),
        )

    def forward(self, X):
        N, h, n, n = X.shape

        X_batched = (
            X.permute(2, 3, 0, 1)
            .contiguous()
            .view(n, n, N * h, 1)
            .permute(2, 3, 0, 1)
            .contiguous()
        )

        if self.training:
            mean = functional.BaryGeom(X_batched)

            with th.no_grad():
                self.running_mean.data = functional.geodesic(
                    self.running_mean, mean, self.momentum
                )

            X_centered = functional.CongrG(X_batched, mean, "neg")

        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, "neg")

        X_normalized = functional.CongrG(X_centered, self.weight, "pos")

        return (
            X_normalized.permute(2, 3, 0, 1)
            .contiguous()
            .view(n, n, N, h)
            .permute(2, 3, 0, 1)
            .contiguous()
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear transformations
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Split into num_heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to value
        attended = torch.matmul(weights, value)

        # Concatenate heads and linearly transform
        concatenated = (
            attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.fc(concatenated)

        return output, weights


class ReEig(nn.Module):
    def forward(self, P):
        return functional.ReEig.apply(P)


class LogEig(nn.Module):
    def forward(self, P):
        return functional.LogEig.apply(P)


class QRDecomposition(nn.Module):
    def forward(self, P):
        return QRFunction.apply(P)
