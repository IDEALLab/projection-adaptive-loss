"""The package-local surrogate architectures import, instantiate and run a forward pass."""

from __future__ import annotations

import torch

from pal.benchmarks.engineering.e1_bwb.bwb_sdf import METADATA, PARAM_ORDER, BWBSDFNet
from pal.benchmarks.engineering.e1_bwb.film_surface import FiLMNet
from pal.benchmarks.engineering.e1_bwb.struct_surrogate import StructuralCVAE


def test_film_net_forward():
    model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                    hidden_dim=64, num_layers=4, extra_layers=3)
    coords = torch.randn(8, 6)
    cond = torch.randn(8, 13)
    out = model(coords, cond)
    assert out.shape == (8, 3)
    assert torch.isfinite(out).all()


def test_structural_cvae_forward():
    model = StructuralCVAE(
        struct_dim=9, bwb_dim=10, thick_dim=3, prop_dim=11,
        latent_dim=10, hidden_dim=32, depth=2,
    )
    struct = torch.randn(4, 9)
    bwb = torch.randn(4, 10)
    thick = torch.randn(4, 3)
    y = torch.randn(4, 1)
    out = model(struct, bwb, thick, y)
    assert out["mu"].shape == (4, 10)
    assert out["log_var"].shape == (4, 10)
    assert out["struct_recon"].shape == (4, 9)
    assert out["properties"].shape == (4, 11)


def test_bwb_sdf_net_forward():
    model = BWBSDFNet(fourier_bands=4, hidden_dim=32, cond_dim=9, n_blocks=2)
    xyz = torch.randn(2, 16, 3)
    cond = torch.randn(2, 9)
    sdf = model(xyz, cond)
    assert sdf.shape == (2, 16)
    assert torch.isfinite(sdf).all()


def test_bwb_sdf_metadata_shape():
    assert METADATA["name"] == "bwb_sdf"
    assert len(PARAM_ORDER) == 9
    assert set(PARAM_ORDER) == {
        "B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3",
    }
