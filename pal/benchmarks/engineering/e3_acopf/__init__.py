"""E3 ACOPF package.

Registration happens in `pal.benchmarks.registry._bootstrap`, one factory
per IEEE case (`e3/acopf_ieee30`, `e3/acopf_ieee57`, `e3/acopf_ieee118`).
"""

from .benchmark import E3ACOPF

__all__ = ["E3ACOPF"]
