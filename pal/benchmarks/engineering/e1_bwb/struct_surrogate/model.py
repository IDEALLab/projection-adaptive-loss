"""Conditional VAE + Predictor for structural surrogate.

  Encoder(struct_9 | bwb_10) -> mu, log_var [d]
  Decoder(z | bwb_10) -> struct_recon [9]
  Predictor(z, bwb_10, thick_3, y) -> properties [11]
  RibPredictor(z, bwb_10, rib_thickness) -> V_ribs [1]
"""

import torch
import torch.nn as nn


def _make_mlp(
    in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float = 0.0,
) -> nn.Sequential:
    """Build a simple MLP: in -> [hidden, GELU, Dropout?] x depth -> out."""
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class StructuralCVAE(nn.Module):
    """Conditional VAE + property predictor + rib volume predictor.

    Args:
        struct_dim: Number of structural params encoded by VAE (9).
        bwb_dim: Number of BWB condition params (10).
        thick_dim: Number of thickness params for predictor (3).
        prop_dim: Number of output properties (11).
        latent_dim: Latent space dimension.
        hidden_dim: Hidden layer width for all components.
        depth: Number of hidden layers for all components.
    """

    def __init__(
        self,
        struct_dim: int = 9,
        bwb_dim: int = 10,
        thick_dim: int = 3,
        prop_dim: int = 11,
        latent_dim: int = 10,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.bwb_dim = bwb_dim
        self.thick_dim = thick_dim
        self.prop_dim = prop_dim
        self.latent_dim = latent_dim

        enc_in = struct_dim + bwb_dim
        self.encoder = _make_mlp(enc_in, 2 * latent_dim, hidden_dim, depth, dropout)

        dec_in = latent_dim + bwb_dim
        self.decoder = _make_mlp(dec_in, struct_dim, hidden_dim, depth, dropout)

        pred_in = latent_dim + bwb_dim + thick_dim + 1  # +1 for y
        self.predictor = _make_mlp(pred_in, prop_dim, hidden_dim, depth, dropout)

        rib_in = latent_dim + bwb_dim + 1  # +1 for rib_thickness
        self.rib_predictor = _make_mlp(rib_in, 1, hidden_dim, depth, dropout)

    def encode(self, struct: torch.Tensor, bwb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode structural params conditioned on BWB.

        Args:
            struct: [B, struct_dim] structural parameters.
            bwb: [B, bwb_dim] BWB condition parameters.

        Returns:
            mu, log_var: [B, latent_dim] each.
        """
        h = self.encoder(torch.cat([struct, bwb], dim=-1))
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * exp(0.5 * log_var)."""
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, bwb: torch.Tensor) -> torch.Tensor:
        """Decode z conditioned on BWB -> structural params.

        Args:
            z: [B, latent_dim]
            bwb: [B, bwb_dim]

        Returns:
            struct_recon: [B, struct_dim]
        """
        return self.decoder(torch.cat([z, bwb], dim=-1))

    def predict(
        self, z: torch.Tensor, bwb: torch.Tensor,
        thick: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Predict structural properties from latent + conditions.

        Args:
            z: [B, latent_dim]
            bwb: [B, bwb_dim]
            thick: [B, thick_dim] (skin_t, front_spar_w, rear_spar_w)
            y: [B, 1] spanwise station

        Returns:
            properties: [B, prop_dim]
        """
        return self.predictor(torch.cat([z, bwb, thick, y], dim=-1))

    def predict_rib_volume(
        self, z: torch.Tensor, bwb: torch.Tensor,
        rib_thickness: torch.Tensor,
    ) -> torch.Tensor:
        """Predict rib volume from latent + BWB + rib thickness.

        Args:
            z: [B, latent_dim]
            bwb: [B, bwb_dim]
            rib_thickness: [B, 1]

        Returns:
            V_ribs: [B, 1]
        """
        return self.rib_predictor(torch.cat([z, bwb, rib_thickness], dim=-1))

    def forward(
        self,
        struct: torch.Tensor,
        bwb: torch.Tensor,
        thick: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode -> sample -> decode + predict.

        Returns dict with: mu, log_var, z, struct_recon, properties.
        """
        mu, log_var = self.encode(struct, bwb)
        z = self.reparameterize(mu, log_var)
        struct_recon = self.decode(z, bwb)
        properties = self.predict(z, bwb, thick, y)
        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "struct_recon": struct_recon,
            "properties": properties,
        }


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), mean over batch."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
