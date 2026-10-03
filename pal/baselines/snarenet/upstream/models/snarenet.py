"""
SnareNet: Neural network with Newton-based projection for hard constraints
"""

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from models.base_model import BaseModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SnareNetRepairLayer(nn.Module):
    """
    Newton-based projection layer for constraint enforcement.

    Projects network outputs onto the constraint manifold using Newton's method
    with the minimum-norm (pseudo-inverse) update step.

    Supports both hard constraints (bl = bu) and inequality constraints (bl < bu),
    with optional adaptive relaxation (bl - eps <= g(y) <= bu + eps).

    `repair` clamps iterates into the box from `data.get_output_box()` when it is not None.
    """

    def __init__(self, data, cfg):
        """
        Args:
            data: Problem data object providing:
                - nineq, neq: int (number of inequality/equality constraints)
                - get_lower_upper_bounds(*inputs) -> (bl, bu)
                - get_g(*inputs) -> callable g(y)
                - get_jacobian(*inputs) -> callable J(y)
            cfg: Config object with cfg.model fields:
                - newton_maxiter, rtol, lambd
        """
        super().__init__()
        self._data = data
        self._maxiter = cfg.model['newton_maxiter']
        self._rtol = cfg.model['rtol']
        self._lambda_reg = cfg.model['lambd']
        self._iter_taken = 0

        self.register_buffer('_eps', torch.zeros(data.nineq + data.neq))

    def set_eps(self, eps):
        """Set epsilon for adaptive relaxation (bl - eps <= g(y) <= bu + eps)."""
        self._eps = eps

    def get_eps(self):
        return self._eps

    def get_iter_taken(self):
        """Number of Newton iterations taken in the last repair call."""
        return self._iter_taken

    def forward(self, output, *constraint_inputs):
        return self.repair(output, *constraint_inputs)

    def repair(self, output, *constraint_inputs):
        """
        Project output to satisfy constraints using Newton's method.

        Args:
            output: Network output tensor (batch_size, n_outputs).
            *constraint_inputs: Passed to constraint methods (e.g. (x,)).

        Returns:
            Projected output satisfying (relaxed) constraints.
        """
        bl, bu = self._data.get_lower_upper_bounds(*constraint_inputs)
        g = self._data.get_g(*constraint_inputs)
        J = self._data.get_jacobian(*constraint_inputs)

        # Clamp iterates into the output box: the surrogates are undefined outside it.
        _get_box = getattr(self._data, "get_output_box", None)
        box = _get_box() if _get_box is not None else None
        if box is not None:
            box = (box[0].to(device=output.device, dtype=output.dtype),
                   box[1].to(device=output.device, dtype=output.dtype))

        y = output
        if box is not None:
            y = torch.clamp(y, min=box[0], max=box[1])

        for i in range(self._maxiter):
            g_y = g(y)
            r_y = -(nn.ReLU()(bl - self._eps - g_y) - nn.ReLU()(g_y - bu - self._eps))

            if torch.amax(torch.abs(r_y)) < self._rtol:
                self._iter_taken = i
                return y

            J_y = J(y)
            dy = self._compute_update_pinv(J_y, r_y)
            y = y - dy
            if box is not None:  # clamp iterate into output box
                y = torch.clamp(y, min=box[0], max=box[1])

            if torch.amax(torch.abs(dy)) < self._rtol:
                self._iter_taken = i + 1
                return y

        self._iter_taken = self._maxiter
        return y

    def _compute_update_pinv(self, J_y, r_y):
        """
        Compute Newton update using pseudo-inverse with optional Tikhonov regularization.

        Args:
            J_y: Jacobian (batch_size, m, n), m = nconstraints, n = output dim.
            r_y: Constraint residual (batch_size, m).

        Returns:
            dy: Newton step (batch_size, n).
        """
        J_yT = J_y.transpose(1, 2)  # (batch_size, n, m)

        if self._lambda_reg > 0:
            # Overdetermined / regularized: solve (J^T J + lambda I) dy = J^T r
            JTJ_reg = torch.bmm(J_yT, J_y)  # (batch_size, n, n)
            JTJ_reg.diagonal(dim1=1, dim2=2).add_(self._lambda_reg)
            rhs = torch.bmm(J_yT, r_y.unsqueeze(-1))  # (batch_size, n, 1)
            dy = torch.linalg.solve(JTJ_reg, rhs).squeeze(-1)
        else:
            # Underdetermined (min-norm): solve J J^T z = r, then dy = J^T z
            JJT = torch.bmm(J_y, J_yT)  # (batch_size, m, m)
            z = torch.linalg.solve(JJT, r_y.unsqueeze(-1))  # (batch_size, m, 1)
            dy = torch.bmm(J_yT, z).squeeze(-1)  # (batch_size, n)

        return dy


class SnareNet(nn.Module):
    """Newton-based neural network for hard-constrained optimization."""

    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._if_repair = True

        self._base = BaseModel(data, cfg)

        self._repair = SnareNetRepairLayer(data, cfg)

    def set_repair(self, val=True):
        self._if_repair = val

    def set_eps(self, eps):
        self._repair.set_eps(eps)

    def get_eps(self):
        return self._repair.get_eps()

    def get_iter_taken(self):
        return self._repair.get_iter_taken()

    def forward(self, x):
        encoded_x = self._data.encode_input(x)
        out = self._base(encoded_x)

        if self._if_repair:
            out = self._repair(out, x)

        return out
