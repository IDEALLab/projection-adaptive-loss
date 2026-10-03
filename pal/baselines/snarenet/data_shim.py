"""PAL Benchmark -> SnareNet problem-class duck type.

PAL `eq` rows map to `b_l = b_u = 0` and `ineq` rows to `b_l = -inf, b_u = 0`.
``model_x = [zeta; conditions]`` feeds the MLP, ``bench_x = conditions`` feeds ``bench.forward``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pal.benchmarks.base import Benchmark


class _SnareNetDataShim:
    """Adapter, PAL Benchmark -> upstream SnareNet `data` object."""

    def __init__(
        self,
        bench: Benchmark,
        model_x: Tensor,
        bench_x: Tensor,
        device: str | torch.device,
        bench_dtype: torch.dtype | None = None,
    ):
        # Follow torch's global default dtype (fp64 under `--precision fp64`).
        if bench_dtype is None:
            bench_dtype = torch.get_default_dtype()
        self.bench = bench
        # [B, condition_dim], possibly [B, 0].
        self._x = bench_x.to(dtype=bench_dtype)
        # [B, zeta_dim + condition_dim].
        self._model_x = model_x.to(dtype=bench_dtype)
        self.device = device
        self._bench_dtype = bench_dtype

        spec = bench.spec
        self.neq = spec.n_eq
        self.nineq = spec.n_ineq
        self.ydim = spec.dim
        self.encoded_xdim = int(model_x.shape[-1])
        self._ct = list(spec.constraint_types)

        # Clamp repair iterates into the output box: the surrogates are undefined outside it.
        if getattr(spec, "hard_output_box", False):
            _lo, _hi = spec.output_bounds
            self._output_box: tuple[Tensor, Tensor] | None = (
                _lo.to(device=self.device), _hi.to(device=self.device),
            )
        else:
            self._output_box = None

        # "loop": K explicit VJPs per Newton iter (visible to probe hooks), "vmap": batched.
        self._jacobian_mode: str = "loop"

    def update_x(self, model_x: Tensor, bench_x: Tensor) -> None:
        """Rebind both inputs; upstream repair fixes `*inputs` at construction."""
        self._model_x = model_x.to(dtype=self._bench_dtype)
        self._x = bench_x.to(dtype=self._bench_dtype)

    def set_jacobian_mode(self, mode: str) -> None:
        """Set the Newton Jacobian construction strategy (``"loop"`` or ``"vmap"``)."""
        if mode not in ("loop", "vmap"):
            raise ValueError(
                f"unknown jacobian_mode {mode!r}; expected 'loop' or 'vmap'"
            )
        self._jacobian_mode = mode

    def encode_input(self, x: Tensor) -> Tensor:
        return x

    def get_lower_upper_bounds(self, *unused):
        K = self.nineq + self.neq
        b_l = torch.zeros(K, device=self.device)
        b_u = torch.zeros(K, device=self.device)
        for i, ct in enumerate(self._ct):
            if ct == "ineq":
                b_l[i] = float("-inf")
        return b_l, b_u

    def _g_batched(self, y: Tensor) -> Tensor:
        # Cast to the benchmark's dtype at the boundary; autograd flows through the cast.
        y_cast = y.to(dtype=self._bench_dtype)
        _, constraints = self.bench.forward(y_cast, self._x)
        out = torch.stack([c.value for c in constraints], dim=-1)     # [B, K]
        return out.to(dtype=y.dtype)

    def _g_single(self, y_b: Tensor, x_b: Tensor) -> Tensor:
        y_cast = y_b.to(dtype=self._bench_dtype).unsqueeze(0)
        x_cast = x_b.to(dtype=self._bench_dtype).unsqueeze(0)
        _, constraints = self.bench.forward(y_cast, x_cast)
        out = torch.stack([c.value[0] for c in constraints])          # [K]
        return out.to(dtype=y_b.dtype)

    def get_output_box(self):
        """Return `(lo, hi)` when the benchmark sets `spec.hard_output_box`, else `None`."""
        return self._output_box

    def get_g(self, *unused):
        return lambda y: self._g_batched(y)

    def get_jacobian(self, *unused):
        """Return `J(y) = dg/dy` evaluated at the current `(y, x)` pair.

        `y` is detached before the Jacobian build, so the Newton update treats `J`
        as a constant (as upstream's `jacrev` does).
        """

        def _j(y: Tensor) -> Tensor:
            # Upstream also calls this under no_grad at eval time.
            with torch.enable_grad():
                if self._jacobian_mode == "loop":
                    y_det = y.detach().requires_grad_(True)
                    g_live = self._g_batched(y_det)  # [B, K]
                    B, K = g_live.shape
                    D = y.shape[-1]
                    J = torch.zeros(B, K, D, device=y.device, dtype=y.dtype)
                    for k in range(K):
                        grad_k = torch.autograd.grad(
                            g_live[:, k].sum(), y_det,
                            retain_graph=(k < K - 1),
                        )[0]
                        J[:, k, :] = grad_k
                else:
                    J = torch.vmap(
                        torch.func.jacrev(self._g_single, argnums=0),
                        in_dims=(0, 0),
                    )(y.detach(), self._x)
            return J.detach()

        return _j

    def evaluate(self, unused_x, y: Tensor) -> Tensor:
        y_cast = y.to(dtype=self._bench_dtype)
        obj, _ = self.bench.forward(y_cast, self._x)
        return obj.to(dtype=y.dtype)                                  # [B]

    def get_resid(self, unused_x, y: Tensor) -> Tensor:
        g = self._g_batched(y)                                        # [B, K]
        b_l, b_u = self.get_lower_upper_bounds()
        return torch.clamp(b_l - g, min=0.0) + torch.clamp(g - b_u, min=0.0)
