"""The "eq" slot stays aligned with ``coverage`` across E2 env configurations.

Env is read at import time, so each configuration runs in a fresh subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE = (
    "from pal.benchmarks.engineering.e2_urban_wind.benchmark import _make_spec\n"
    "s = _make_spec()\n"
    "assert s.constraint_types.count('eq') == s.n_eq\n"
    "assert s.n_eq + s.n_ineq == len(s.constraint_names)\n"
    "cov = [i for i, n in enumerate(s.constraint_names) if n == 'coverage']\n"
    "for i, t in enumerate(s.constraint_types):\n"
    "    assert (t == 'eq') == (i in cov), (i, t, cov)\n"
    "print(s.n_eq, s.n_ineq, len(s.constraint_names))\n"
)


def _run(env_overrides: dict[str, str]) -> str:
    import os

    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"probe failed for {env_overrides}:\n{proc.stdout}\n{proc.stderr}"
    )
    return proc.stdout.strip().splitlines()[-1]


class CoverageEqLayoutTests(unittest.TestCase):
    def test_default_per_pair_off(self):
        # site, clearance, height, danger, coverage(eq) => 1 eq + 4 ineq.
        out = _run({"E2_CLEARANCE_PER_PAIR": "0"})
        self.assertEqual(out, "1 4 5")

    def test_per_pair_on(self):
        # 45 per-pair clearance ineqs + site + height + danger + coverage(eq).
        out = _run({"E2_CLEARANCE_PER_PAIR": "1"})
        self.assertEqual(out, "1 48 49")

    def test_dropped_constraints_keep_coverage_eq(self):
        # Dropping two ineqs leaves coverage as the sole eq.
        out = _run(
            {
                "E2_CLEARANCE_PER_PAIR": "0",
                "E2_DROP_CONSTRAINTS": "clearance,height",
            }
        )
        self.assertEqual(out, "1 2 3")

    def test_dropping_coverage_yields_no_eq(self):
        out = _run(
            {
                "E2_CLEARANCE_PER_PAIR": "0",
                "E2_DROP_CONSTRAINTS": "coverage",
            }
        )
        self.assertEqual(out, "0 4 4")


if __name__ == "__main__":
    unittest.main()
