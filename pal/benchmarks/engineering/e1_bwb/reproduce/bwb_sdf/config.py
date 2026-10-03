"""Default hyperparameters for BWB SDF training."""

DEFAULT_CONFIG = {
    "batch_size": 32,           # geometries per step
    "points_per_geom": 4096,    # subsampled from 30k stored

    "fourier_bands": 10,
    "hidden_dim": 512,
    "cond_dim": 9,              # geom params

    "lr": 1e-4,
    "total_steps": 500_000,
    "lambda_eik": 0.1,
    "eik_subsample": 0.25,
    "grad_clip": 1.0,

    "log_interval": 200,
    "vis_interval": 5000,
    "ckpt_interval": 10_000,
    "output_dir": "runs/bwb_sdf",
}
