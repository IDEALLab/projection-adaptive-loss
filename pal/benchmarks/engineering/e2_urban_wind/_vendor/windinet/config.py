# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: added ScalarConditioningConfig, removed LoRA and text conditioning.
"""Pydantic configuration models for WinDiNet training and inference."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

try:
    from windinet.inference.model_loader import LtxvModelVersion
except ImportError:
    LtxvModelVersion = None


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model."""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class ConditioningConfig(ConfigBaseModel):
    """Configuration for conditioning during training."""

    first_frame_conditioning_p: float = Field(default=0.1, ge=0.0, le=1.0)


class ScalarConditioningConfig(ConfigBaseModel):
    """Configuration for scalar embeddings (e.g., inlet_speed, field_size)."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable scalar conditioning",
    )

    scalar_names: list[str] = Field(
        default=["inlet_speed_mps", "field_size_m"],
        description="Names of scalars to embed",
    )

    scalar_ranges: dict[str, tuple[float, float]] = Field(
        default={
            "inlet_speed_mps": (0.1, 20.0),
            "field_size_m": (900.0, 1400.0),
        },
        description="Min/max ranges for each scalar (used for normalization to [0, 1])",
    )

    embedding_dim: int = Field(
        default=4096,
        description="Dimension of scalar embeddings (should match transformer hidden size)",
    )

    num_tokens_per_scalar: int = Field(
        default=4,
        description="Number of embedding tokens to generate per scalar",
        ge=1,
        le=16,
    )

    fourier_features: int = Field(
        default=64,
        description="Number of Fourier features for positional encoding of scalars",
        ge=8,
    )

    mlp_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension of the MLP that processes scalar embeddings",
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for scalar embedding MLP",
        ge=0.0,
        le=1.0,
    )

    @field_validator("scalar_ranges")
    @classmethod
    def validate_scalar_ranges(cls, v: dict, info: ValidationInfo) -> dict:
        """Validate that ranges are provided for all scalar names."""
        scalar_names = info.data.get("scalar_names", [])
        for name in scalar_names:
            if name not in v:
                raise ValueError(f"Range must be provided for scalar '{name}'")
            min_val, max_val = v[name]
            if min_val >= max_val:
                raise ValueError(f"Invalid range for scalar '{name}': min ({min_val}) >= max ({max_val})")
        return v


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters."""

    learning_rate: float = Field(default=5e-4)
    steps: int = Field(default=3000)
    batch_size: int = Field(default=2)
    gradient_accumulation_steps: int = Field(default=1)
    max_grad_norm: float = Field(default=1.0)
    optimizer_type: Literal["adamw", "adamw8bit"] = Field(default="adamw")
    scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"] = Field(
        default="linear"
    )
    scheduler_params: dict = Field(default_factory=dict)
    enable_gradient_checkpointing: bool = Field(default=False)


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization."""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(default="bf16")
    compile_with_inductor: bool = Field(default=True)
    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(default="reduce-overhead")


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing."""

    preprocessed_data_root: str = Field(description="Path to folder containing preprocessed training data")
    num_dataloader_workers: int = Field(default=2, ge=0)


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training."""

    interval: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=1, ge=-1)


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(default=False)
    project: str = Field(default="windinet")
    entity: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training."""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(default="shifted_logit_normal")
    timestep_sampling_params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# VAE decoder finetuning configs
# ---------------------------------------------------------------------------


class PhysicsLossConfig(ConfigBaseModel):
    """Physics-informed loss weights for VAE decoder finetuning."""

    distance_alpha: float = Field(default=2.0, description="Wall proximity weight amplification")
    distance_sigma: float = Field(default=20.0, description="Gaussian falloff distance for wall weighting (pixels)")
    lambda_div: float = Field(default=10.0, description="Divergence (incompressibility) loss weight")
    lambda_wall: float = Field(default=10.0, description="Wall no-penetration loss weight")
    wall_band: int = Field(default=2, description="Wall mask dilation in pixels")
    warmup_frames: int = Field(default=56, description="Physics losses only apply after this many frames")


class VaeDataConfig(ConfigBaseModel):
    """Dataset configuration for VAE decoder finetuning."""

    data_root: str = Field(description="Path to wind simulation data (root/<id>/fields.npz)")
    wind_norm: float = Field(default=30.0, description="Wind speed normalisation constant (m/s)")
    eval_sims: int = Field(default=10, description="Number of simulations held out for VRMSE evaluation")
    num_dataloader_workers: int = Field(default=4, ge=0)


class VaeOptimizationConfig(ConfigBaseModel):
    """Optimization for VAE decoder finetuning."""

    learning_rate: float = Field(default=5e-5)
    min_learning_rate: float = Field(default=1e-6, description="Minimum LR for cosine annealing")
    epochs: int = Field(default=10)
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=32)
    max_grad_norm: float = Field(default=5.0)
    weight_decay: float = Field(default=0.0)
    warmup_steps: int = Field(default=50, description="Linear warmup optimizer steps")
    warmup_start_factor: float = Field(default=0.01)
    vis_every_steps: int = Field(default=0, description="Visualize GT|Reconstruction|Error every N optimizer steps (0=disabled)")


class VaeTrainerConfig(ConfigBaseModel):
    """Configuration for VAE decoder finetuning with physics-informed losses."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: VaeDataConfig = Field(default_factory=VaeDataConfig)
    optimization: VaeOptimizationConfig = Field(default_factory=VaeOptimizationConfig)
    loss: PhysicsLossConfig = Field(default_factory=PhysicsLossConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs/vae_finetune")

    @field_validator("output_dir")
    @classmethod
    def expand_vae_output_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


# ---------------------------------------------------------------------------
# Diffusion model training configs
# ---------------------------------------------------------------------------


class LtxvTrainerConfig(ConfigBaseModel):
    """Unified configuration for WinDiNet training."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    scalar_conditioning: ScalarConditioningConfig = Field(default_factory=ScalarConditioningConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())
