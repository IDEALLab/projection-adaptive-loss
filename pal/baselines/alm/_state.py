"""ALM state for Basir & Senocak (arXiv:2306.04904v2), Algorithm 3 (adaptive penalty updates)."""

from __future__ import annotations

import torch
from torch import Tensor


class ALMState:
    """Per-constraint (lambda, mu, v_bar) following paper Algorithm 3.

    Paper defaults (Algorithm 3 line 1):  gamma = 1e-2, alpha = 0.99, eps = 1e-8.
    Paper init    (Algorithm 3 lines 3-5): lambda^0 = 1, mu^0 = 1, v_bar^0 = 0.
    """

    def __init__(
        self,
        n_constraints: int,
        constraint_types: list[str],
        device: str,
        gamma: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        mu_init: float = 1.0,
        lambda_init: float = 1.0,
    ):
        assert len(constraint_types) == n_constraints
        self.n = n_constraints
        self.types = list(constraint_types)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

        # Algorithm 3, lines 3-5
        self.lambdas = torch.full((n_constraints,), lambda_init, device=device)
        self.mu = torch.full((n_constraints,), mu_init, device=device)
        self.v = torch.zeros(n_constraints, device=device)

    def compute_loss(self, constraints: Tensor) -> Tensor:
        """Augmented Lagrangian loss term, paper eq. (5):
            L_pen = sum_i lambda_i * C_i + (mu_i/2) * C_i^2

        `constraints`: [B, K] residuals, ineq slots already clamped to ``>= 0``.
        """
        lam = self.lambdas.detach()
        mu = self.mu.detach()
        return (lam * constraints + (mu / 2) * constraints.pow(2)).sum(dim=1)

    def update(self, constraints: Tensor) -> None:
        """Apply paper Algorithm 3 lines 8-10 after the primal step.

        `constraints`: [B, K] detached signed residuals (with ineq slots
        already non-negative via clamp_min(0) in the caller).
        """
        with torch.no_grad():
            # Eq. (7): EMA of squared C, with C averaged over the minibatch.
            c_mean = constraints.detach().mean(dim=0)  # [K]
            self.v = self.alpha * self.v + (1.0 - self.alpha) * c_mean.pow(2)

            # Eq. (8): eps inside the sqrt caps mu at gamma/sqrt(eps) when v_bar = 0.
            self.mu = self.gamma / torch.sqrt(self.v + self.eps)

            # Eq. (9): dual update.
            self.lambdas = self.lambdas + self.mu * c_mean

    def log_dict(self) -> dict:
        d: dict[str, float] = {}
        for k in range(self.n):
            d[f"alm/lambda_{k}"] = self.lambdas[k].item()
            d[f"alm/mu_{k}"] = self.mu[k].item()
            d[f"alm/v_{k}"] = self.v[k].item()
        return d

    def state_dict(self) -> dict:
        return {
            "lambdas": self.lambdas.clone(),
            "mu": self.mu.clone(),
            "v": self.v.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore (lambda, mu, v_bar) from `state_dict()`, onto the current device."""
        device = self.lambdas.device
        self.lambdas = state["lambdas"].detach().clone().to(device)
        self.mu = state["mu"].detach().clone().to(device)
        self.v = state["v"].detach().clone().to(device)
