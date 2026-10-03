"""PAL-SQP: PAL-LogGap with the elastic-mode SQP repair step.

Identical to `pal_loggap` except that `Projector.method` is `"sqp"`, the
per-sample elastic feasibility QP solved by OSQP through `qpsolvers`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pal.method.loggap.solver import PALLogGapConfig, PALLogGapSolver
from pal.projection.projector import _SQP_RHO


@dataclass
class PALSqpConfig(PALLogGapConfig):
    """`PALLogGapConfig` with the SQP repair step selected.

    `proj_sqp_rho` is the only exposed knob; the QP's numerical settings stay
    on `Projector`.
    """

    proj_method: str = "sqp"
    # Elastic penalty rho in `min 1/2||Delta||^2 + rho||s||1`.
    proj_sqp_rho: float = _SQP_RHO


class PALSqpSolver(PALLogGapSolver):
    """`PALLogGapSolver` with `proj_method="sqp"`. Training loop is inherited."""

    name = "pal_sqp"

    def __init__(self, config: PALSqpConfig | None = None):
        self.config = config or PALSqpConfig()
