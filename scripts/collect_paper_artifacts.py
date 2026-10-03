#!/usr/bin/env python3
"""Copy only the small paper-facing files from a rendered run root.

Example:
    python scripts/collect_paper_artifacts.py \
        --source $SCRATCH/pal-runs/e2_paper \
        --label 2026-04-23_e2_paper \
        --dest results/gpu_exports
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PATTERNS = [
    "results.md",
    "paper_results/paper_results.md",
    "paper_results/paper_results_summary.csv",
    "paper_results/paper_results_gap_summary.csv",
    "paper_results/latex/*.tex",
]


def _copy_if_small(src: Path, dst: Path, *, max_bytes: int) -> tuple[bool, str]:
    if not src.exists():
        return False, "missing"
    size = src.stat().st_size
    if size > max_bytes:
        return False, f"skipped_too_large:{size}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True, f"copied:{size}"


def _rel_to_source(src_root: Path, path: Path) -> str:
    return str(path.relative_to(src_root))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="collect lightweight paper artifacts into the repo")
    p.add_argument("--source", required=True, help="rendered run root on scratch")
    p.add_argument("--label", required=True, help="folder name under --dest")
    p.add_argument(
        "--dest",
        default="results/gpu_exports",
        help="repo-local destination root (default: results/gpu_exports)",
    )
    p.add_argument(
        "--max-mb",
        type=float,
        default=2.0,
        help="skip any single file above this size in MB (default: 2.0)",
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="extra glob relative to --source; may be passed multiple times",
    )
    args = p.parse_args(argv)

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"[collect] source does not exist: {source}")

    repo_root = Path(__file__).resolve().parents[1]
    dest_root = Path(args.dest)
    if not dest_root.is_absolute():
        dest_root = (repo_root / dest_root).resolve()
    out_dir = dest_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(float(args.max_mb) * 1024 * 1024)
    patterns = [*DEFAULT_PATTERNS, *args.include]

    copied: list[dict[str, str | int]] = []
    skipped: list[dict[str, str | int]] = []
    seen: set[Path] = set()

    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if path.is_dir() or path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(source)
            dst = out_dir / rel
            ok, reason = _copy_if_small(path, dst, max_bytes=max_bytes)
            entry = {
                "path": str(rel),
                "reason": reason,
                "bytes": int(path.stat().st_size) if path.exists() else 0,
            }
            if ok:
                copied.append(entry)
                print(f"[collect] copied {rel}")
            else:
                skipped.append(entry)
                print(f"[collect] skipped {rel} ({reason})")

    manifest = {
        "source": str(source),
        "label": args.label,
        "collected_at": datetime.now(UTC).isoformat(),
        "max_bytes": max_bytes,
        "copied": copied,
        "skipped": skipped,
    }
    manifest_path = out_dir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[collect] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
