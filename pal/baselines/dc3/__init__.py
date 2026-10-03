"""DC3 baseline, Donti, Rolnick & Kolter, ICLR 2021 (arXiv:2104.12225).

Adapter only. The algorithmic core (``grad_steps``, ``grad_steps_all``,
``total_loss``) is loaded unmodified from ``upstream/method.py`` via
``_upstream_loader``. Completion, partial-correction gradient, and
backbone integration are PAL-authored.
"""

from pal.baselines.dc3.solver import DC3Config, DC3Solver

__all__ = ["DC3Config", "DC3Solver"]
