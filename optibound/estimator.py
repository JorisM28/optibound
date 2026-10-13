import math
import numpy as np
import torch


def euclidian_distance(X, Y):
    return torch.linalg.vector_norm((X - Y).float(), dim=-1)


class MonteCarloLipschitz:

    def __init__(self, func, domain, n_samples, device="cpu"):
        self.func = func
        self.low, self.high = domain
        self.n_samples = n_samples
        self.device = device

    def estimate(self, batch_size, epsilon=1e-7, k=2.0):
        n_iter = math.floor(self.n_samples / batch_size)

        sup_R = 0.0
        sum_R = 0.0
        sum_sq_R = 0.0
        N_valid = 0

        for _ in range(n_iter):
            shape = (batch_size, self.func.dimension)

            X_batch = (
                self.low
                + (self.high - self.low) * torch.rand(shape, device=self.device)
            )
            Y_batch = (
                self.low
                + (self.high - self.low) * torch.rand(shape, device=self.device)
            )

            D = euclidian_distance(X_batch, Y_batch)

            mask = D >= epsilon
            if not mask.any():
                continue

            D_valid = D[mask]
            X_valid = X_batch[mask]
            Y_valid = Y_batch[mask]

            X_grad = self.func.gradient(X_valid)
            Y_grad = self.func.gradient(Y_valid)

            G = euclidian_distance(X_grad, Y_grad)
            R = G / D_valid


            batch_max = torch.max(R).item()
            sup_R = max(sup_R, batch_max)

            sum_R += torch.sum(R).item()
            sum_sq_R += torch.sum(R**2).item()
            N_valid += D_valid.numel()

        if N_valid == 0:
            raise RuntimeError(
                "None valid pair found. Increase the domain or reduce" \
                "epsilon."
            )

        mean_R = sum_R / N_valid
        var_R = max(0.0, (sum_sq_R / N_valid) - (mean_R**2))
        std_R = math.sqrt(var_R)


        L_hat = sup_R + k * std_R

        return {
            "L_sup": sup_R,
            "L_hat": L_hat,
            "mean": mean_R,
            "std": std_R,
            "valid_samples": N_valid,
            "suggested_lr": 1.0 / L_hat,
        }