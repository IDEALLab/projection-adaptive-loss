"""PAL-IP: PAL-LogGap with the interior-point (barrier-Newton) repair step.

Identical to `pal_loggap` except that `Projector.method` is `"ip"`: one damped
Newton step on the log-barrier KKT system of the linearized feasibility problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from pal.method.loggap.solver import PALLogGapConfig, PALLogGapSolver
from pal.projection.projector import _IP_MU0


@dataclass
class PALIpConfig(PALLogGapConfig):
    """`PALLogGapConfig` with the interior-point repair step selected.

    `proj_ip_mu0` is the only exposed knob (`mu = mu0 * v(y_hat)` with `v` the inf-norm
    violation). `proj_ip_fixed_mu` replaces that rule with a constant mu.
    """

    proj_method: str = "ip"
    # mu0 in `mu = mu0 * v(y_hat)`.
    proj_ip_mu0: float = _IP_MU0
    # Constant mu per sample and inference iteration; None uses the rule above.
    proj_ip_fixed_mu: float | None = None


class PALIpSolver(PALLogGapSolver):
    """`PALLogGapSolver` with `proj_method="ip"`. Training loop is inherited."""

    name = "pal_ip"

    def __init__(self, config: PALIpConfig | None = None):
        self.config = config or PALIpConfig()
