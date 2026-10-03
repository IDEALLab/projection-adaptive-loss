"""Smoke tests for pal.utils.peak_memory.probe_peak_memory."""

from __future__ import annotations

import pytest
import torch

from pal.utils.peak_memory import probe_peak_memory


def test_disabled_is_noop_returns_zero(monkeypatch):
    monkeypatch.delenv("PAL_MEM_PROBE", raising=False)
    with probe_peak_memory(enabled=False) as r:
        x = torch.zeros(1024)  # noqa: F841
    assert r["enabled"] is False
    assert r["peak_bytes"] == 0
    assert r["n_devices"] == 0


def test_cpu_returns_zero_when_no_cuda(monkeypatch):
    monkeypatch.delenv("PAL_MEM_PROBE", raising=False)
    if torch.cuda.is_available():
        pytest.skip("CUDA available, covered by CUDA test")
    with probe_peak_memory(enabled=True) as r:
        x = torch.zeros(1024)  # noqa: F841
    assert r["enabled"] is False  # gracefully downgraded
    assert r["peak_bytes"] == 0
    assert r["n_devices"] == 0


def test_rss_mode_records_nonzero_peak(monkeypatch):
    monkeypatch.setenv("PAL_MEM_PROBE", "rss")
    with probe_peak_memory(enabled=True) as r:
        big = torch.zeros(8 * 1024 * 1024, dtype=torch.float64)  # 64 MB
        s = big.sum().item()
        del big, s
    assert r["enabled"] is True
    assert r["mode"] == "rss"
    assert r["peak_bytes"] > 1024 * 1024  # >1 MB sanity floor


def test_rss_mode_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("PAL_MEM_PROBE", "rss")
    with probe_peak_memory(enabled=False) as r:
        x = torch.zeros(1024)  # noqa: F841
    assert r["enabled"] is False
    assert r["peak_bytes"] == 0


def test_dict_is_mutable_and_readable_after_block():
    with probe_peak_memory(enabled=False) as r:
        assert r["peak_bytes"] == 0
        r["sentinel"] = 42
    assert r["sentinel"] == 42
    assert "peak_bytes" in r


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_records_nonzero_peak():
    with probe_peak_memory(enabled=True) as r:
        x = torch.zeros(1024 * 1024, device="cuda")
        del x
    assert r["enabled"] is True
    assert r["peak_bytes"] > 0
    assert r["n_devices"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_resets_between_blocks():
    with probe_peak_memory(enabled=True) as r1:
        x = torch.zeros(4 * 1024 * 1024, device="cuda")
        del x
    big = r1["peak_bytes"]

    with probe_peak_memory(enabled=True) as r2:
        x = torch.zeros(1024, device="cuda")
        del x
    small = r2["peak_bytes"]

    assert big > small, f"reset_peak_memory_stats should have cleared between blocks: {big=} {small=}"
