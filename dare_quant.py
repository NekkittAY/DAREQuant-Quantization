import torch
import torch.nn as nn
import numpy as np
import math


class DAREQuantLinear(nn.Module):
    def __init__(self, linear, bits=4, bins=128, lambda_weight=1.0, lambda_entropy=0.1):
        super().__init__()
        self.bits = bits
        self.bins = bins
        self.lambda_weight = lambda_weight
        self.lambda_entropy = lambda_entropy
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

        W = linear.weight.data.clone()
        self.register_buffer("weight_q", self.quantize(W))

    def quantize(self, W):
        qmin = -(2 ** (self.bits - 1))
        qmax = (2 ** (self.bits - 1)) - 1

        Wq = torch.zeros_like(W)
        for c in range(W.size(0)):
            w = W[c]
            best_scale, best_loss = None, float("inf")

            for scale in torch.linspace(w.abs().max()/qmax, w.abs().max(), steps=20):
                qw = torch.clamp((w / scale).round(), qmin, qmax) * scale

                residual = (w - qw).cpu().numpy()
                hist, _ = np.histogram(residual, bins=self.bins, density=True)
                hist = hist + 1e-8
                entropy = -np.sum(hist * np.log(hist))

                l2_error = ((w - qw)**2).mean().item()

                loss = self.lambda_entropy * entropy + self.lambda_weight * l2_error

                if loss < best_loss:
                    best_loss = loss
                    best_scale = scale

            Wq[c] = torch.clamp((w / best_scale).round(), qmin, qmax) * best_scale

        return Wq

    def forward(self, x):
        return nn.functional.linear(x, self.weight_q, self.bias)
