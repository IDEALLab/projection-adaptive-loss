from __future__ import annotations

from pal.artifacts.core import ArtifactRef

# Files <= 10 MB ship in the repo (`location="in_repo"`), larger ones on Hugging Face.
REGISTRY: dict[str, list[ArtifactRef]] = {
    "e2/urban_wind": [
        ArtifactRef(
            name="scalar_embedding",
            hf_filename="scalar_embedding.safetensors",
            hf_revision="66e8c7eba1735e0a3fdcf17b704a3a672abdc531",
            local_subdir="windinet",
            location="hf",
        ),
        ArtifactRef(
            name="vae_decoder",
            hf_filename="vae_decoder.safetensors",
            hf_revision="66e8c7eba1735e0a3fdcf17b704a3a672abdc531",
            local_subdir="windinet",
            location="hf",
        ),
        ArtifactRef(
            name="dit",
            hf_filename="dit.safetensors",
            hf_revision="66e8c7eba1735e0a3fdcf17b704a3a672abdc531",
            local_subdir="windinet",
            location="hf",
        ),
    ],
    "e1/bwb": [
        ArtifactRef(
            name="a_aero_weights",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/a_aero/a_aero.pt",
        ),
        ArtifactRef(
            name="a_aero_norm_stats",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/a_aero/a_aero_norm_stats.json",
        ),
        ArtifactRef(
            name="film_weights",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/film_surface/film_ep2478_val0.056.pth",
        ),
        ArtifactRef(
            name="film_norm_stats",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/film_surface/norm_stats.json",
        ),
        ArtifactRef(
            name="struct_weights",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/struct/best.pt",
        ),
        ArtifactRef(
            name="bwb_sdf_weights",
            location="in_repo",
            repo_path="pal/benchmarks/engineering/e1_bwb/_artifacts_data/bwb_sdf/adamw_2_march.pt",
        ),
    ],
}


def ref_by_name(bench_id: str, name: str) -> ArtifactRef:
    """Look up a registered artifact by its logical name.

    Raises `KeyError` if the bench or name is unknown.
    """
    refs = REGISTRY.get(bench_id)
    if refs is None:
        known = ", ".join(sorted(REGISTRY)) or "(empty)"
        raise KeyError(f"no artifacts registered for bench {bench_id!r}; known: {known}")
    for ref in refs:
        if ref.name == name:
            return ref
    available = ", ".join(r.name for r in refs) or "(none)"
    raise KeyError(
        f"no artifact named {name!r} for bench {bench_id!r}; available: {available}"
    )
