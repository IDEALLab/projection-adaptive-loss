"""Replica monkeypatch rebinding for E2's ParallelSurrogate."""

from __future__ import annotations

import pytest
import torch

import pal.benchmarks.engineering.e2_urban_wind.parallel_surrogate as e2_parallel


def test_e2_parallel_surrogate_rebinds_replica_monkeypatches(monkeypatch) -> None:
    # Like torch.utils.checkpoint: absorb use_reentrant, forward everything else.
    monkeypatch.setattr(
        e2_parallel,
        "checkpoint",
        lambda fn, *args, use_reentrant=False, **kwargs: fn(*args, **kwargs),
    )

    class FakeResnet(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.marker = float(marker)

        def forward(self, x, temb=None, generator=None):
            return x + self.marker

    class FakeMidBlock(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.resnets = torch.nn.ModuleList([FakeResnet(marker)])

    class FakeUpBlock(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.resnets = torch.nn.ModuleList([FakeResnet(marker)])

    class FakeDecoder(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.mid_block = FakeMidBlock(marker)
            self.up_blocks = torch.nn.ModuleList([FakeUpBlock(marker + 10.0)])

    class FakeVAE(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.marker = float(marker)
            self.decoder = FakeDecoder(marker)
            self.proj = torch.nn.Linear(1, 1)

    class FakeAdaptedVAE(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.vae = FakeVAE(marker)

    class FakeSurrogate(torch.nn.Module):
        def __init__(self, marker: float):
            super().__init__()
            self.marker = float(marker)
            self.adapted_vae = FakeAdaptedVAE(marker)
            self.proj = torch.nn.Linear(1, 1)

        def forward(self, building_mask, inlet_u, inlet_v, seed=42):
            return building_mask, inlet_u

    base = FakeSurrogate(marker=1.0)

    for resnet in base.adapted_vae.vae.decoder.mid_block.resnets:
        original_forward = resnet.forward

        def ckpt_forward(*args, original_forward=original_forward, **kwargs):
            return e2_parallel.checkpoint(
                original_forward, *args, use_reentrant=False, **kwargs
            )

        resnet.forward = ckpt_forward

    for block in base.adapted_vae.vae.decoder.up_blocks:
        for resnet in block.resnets:
            original_forward = resnet.forward

            def ckpt_forward(*args, original_forward=original_forward, **kwargs):
                return e2_parallel.checkpoint(
                    original_forward, *args, use_reentrant=False, **kwargs
                )

            resnet.forward = ckpt_forward

    parallel = e2_parallel.ParallelSurrogate(base, devices=["cpu", "cpu"])
    replica = parallel.replicas[1]
    parallel.replicas[0].adapted_vae.vae.decoder.mid_block.resnets[0].marker = 100.0
    parallel.replicas[0].adapted_vae.vae.decoder.up_blocks[0].resnets[0].marker = 110.0

    x = torch.tensor([2.0], requires_grad=True)
    mid_out = replica.adapted_vae.vae.decoder.mid_block.resnets[0](x)
    up_out = replica.adapted_vae.vae.decoder.up_blocks[0].resnets[0](x)
    assert mid_out.item() == pytest.approx(3.0)
    assert up_out.item() == pytest.approx(13.0)
