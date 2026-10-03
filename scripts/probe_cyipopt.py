#!/usr/bin/env python3
"""Tiny cyipopt smoke test used by the GH200 IPOPT container path."""

from __future__ import annotations

import cyipopt
import numpy as np


class _Quadratic:
    def objective(self, x):
        return float((x[0] - 3.0) ** 2)

    def gradient(self, x):
        return np.array([2.0 * (x[0] - 3.0)], dtype=np.float64)

    def constraints(self, x):
        return np.array([], dtype=np.float64)

    def jacobian(self, x):
        return np.array([], dtype=np.float64)


def main() -> int:
    nlp = cyipopt.Problem(
        n=1,
        m=0,
        problem_obj=_Quadratic(),
        lb=np.array([-10.0], dtype=np.float64),
        ub=np.array([10.0], dtype=np.float64),
        cl=np.array([], dtype=np.float64),
        cu=np.array([], dtype=np.float64),
    )
    x_opt, info = nlp.solve(np.array([0.0], dtype=np.float64))
    print(f"x*={x_opt[0]:.6f} status={info['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
