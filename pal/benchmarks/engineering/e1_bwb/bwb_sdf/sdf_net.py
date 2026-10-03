"""Conditional SDF network for BlendedNet BWB geometries.

Maps (x, y, z, geom_params) -> signed distance.
Two variants: Fourier+GELU (DeepSDF-style) or SIREN (sin activations).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class FourierEncoder(nn.Module):
    """NeRF-style Fourier positional encoding.

    Args:
        n_bands: number of frequency bands (L)
        input_dim: dimension of input coordinates
    """

    def __init__(self, n_bands: int = 10, input_dim: int = 3):
        super().__init__()
        self.n_bands = n_bands
        self.input_dim = input_dim
        self.output_dim = input_dim * (2 * n_bands + 1)
        freqs = 2.0 ** torch.arange(n_bands).float()
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        """Encode spatial coordinates.

        Args:
            x: [..., input_dim] coordinates

        Returns:
            [..., output_dim] Fourier features
        """
        scaled = x.unsqueeze(-1) * self.freqs * math.pi
        sin_feat = scaled.sin().flatten(-2)
        cos_feat = scaled.cos().flatten(-2)
        return torch.cat([x, sin_feat, cos_feat], dim=-1)


class SirenLayer(nn.Module):
    """Single SIREN layer: Linear + sin activation with omega_0 scaling.

    Args:
        in_dim: input dimension
        out_dim: output dimension
        omega_0: frequency scaling factor
        is_first: if True, use special first-layer init
    """

    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)

        # SIREN-specific init
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = math.sqrt(6.0 / in_dim) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class BWBSDFNet(nn.Module):
    """Conditional SDF network for BWB shapes.

    Two modes controlled by `activation`:
    - "gelu": Fourier(xyz) + GELU MLP (DeepSDF-style)
    - "siren": raw xyz + sin MLP (SIREN-style, no Fourier needed)

    Both use a skip connection at the midpoint.

    Args:
        fourier_bands: number of Fourier frequency bands (ignored for siren)
        hidden_dim: width of hidden layers
        cond_dim: dimension of condition vector (geom params)
        n_blocks: layers per block (total depth = 2*n_blocks)
        activation: "gelu" or "siren"
        omega_0: SIREN frequency scaling (only for siren)
    """

    def __init__(
        self,
        fourier_bands: int = 10,
        hidden_dim: int = 512,
        cond_dim: int = 9,
        n_blocks: int = 3,
        activation: str = "gelu",
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.activation = activation

        if activation == "siren":
            # SIREN: raw xyz (no Fourier encoding)
            input_dim = 3 + cond_dim
            self.encoder = None

            layers1 = [SirenLayer(input_dim, hidden_dim, omega_0, is_first=True)]
            for _ in range(n_blocks - 1):
                layers1.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
            self.block1 = nn.Sequential(*layers1)

            self.block2_in = SirenLayer(hidden_dim + input_dim, hidden_dim, omega_0)
            layers2 = []
            for _ in range(n_blocks - 1):
                layers2.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
            self.block2 = nn.Sequential(*layers2)

            self.head = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / omega_0
                self.head.weight.uniform_(-bound, bound)

        else:
            # Fourier + GELU
            self.encoder = FourierEncoder(n_bands=fourier_bands, input_dim=3)
            input_dim = self.encoder.output_dim + cond_dim

            layers1 = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            for _ in range(n_blocks - 1):
                layers1.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            self.block1 = nn.Sequential(*layers1)

            self.block2_in = nn.Linear(hidden_dim + input_dim, hidden_dim)
            layers2 = [nn.GELU()]
            for _ in range(n_blocks - 1):
                layers2.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            self.block2 = nn.Sequential(*layers2)
            self.head = nn.Linear(hidden_dim, 1)

            self._init_weights_gelu()

    def _init_weights_gelu(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz: Tensor, cond: Tensor) -> Tensor:
        """Forward pass.

        Args:
            xyz: [B, Q, 3] spatial coordinates
            cond: [B, 9] normalized condition vector (geom params)

        Returns:
            [B, Q] SDF values
        """
        B, Q, _ = xyz.shape

        # enforce Y-symmetry: SDF(x,y,z) = SDF(x,-y,z)
        xyz = torch.cat([xyz[..., :1], xyz[..., 1:2].abs(), xyz[..., 2:]], dim=-1)

        if self.encoder is not None:
            feat_xyz = self.encoder(xyz)
        else:
            feat_xyz = xyz  # SIREN uses raw coords

        cond_exp = cond.unsqueeze(1).expand(B, Q, -1)

        x = torch.cat([feat_xyz, cond_exp], dim=-1)
        inp = x

        x = self.block1(x)

        x = torch.cat([x, inp], dim=-1)
        x = self.block2_in(x)
        x = self.block2(x)

        sdf = self.head(x).squeeze(-1)
        return sdf
