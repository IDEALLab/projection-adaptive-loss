"""Tests for the DC3 upstream loader: exports, dtype isolation, sys.modules cleanup."""

from __future__ import annotations

import sys
import unittest

import torch


class UpstreamLoaderTests(unittest.TestCase):
    def test_returns_callables(self):
        from pal.baselines.dc3._upstream_loader import load_vendored
        gs, gsa, tl = load_vendored()
        self.assertTrue(callable(gs))
        self.assertTrue(callable(gsa))
        self.assertTrue(callable(tl))

    def test_dtype_preserved(self):
        from pal.baselines.dc3._upstream_loader import load_vendored
        prev = torch.get_default_dtype()
        load_vendored()
        self.assertEqual(torch.get_default_dtype(), prev)

    def test_stubs_scrubbed(self):
        from pal.baselines.dc3._upstream_loader import load_vendored
        for k in ("utils", "default_args", "setproctitle"):
            sys.modules.pop(k, None)
        load_vendored()
        for k in ("utils", "default_args", "setproctitle"):
            self.assertNotIn(k, sys.modules, f"{k} leaked into sys.modules")

    def test_method_module_scrubbed(self):
        from pal.baselines.dc3._upstream_loader import load_vendored
        load_vendored()
        self.assertNotIn(
            "pal.baselines.dc3.upstream.method", sys.modules,
            "vendored method module leaked into sys.modules",
        )

    def test_idempotent(self):
        from pal.baselines.dc3._upstream_loader import load_vendored
        load_vendored()
        load_vendored()  # must not error


if __name__ == "__main__":
    unittest.main()
