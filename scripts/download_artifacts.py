"""Download PAL benchmark artifacts from the Hugging Face repo (`PAL_HF_REPO_ID`).

Usage:
    python scripts/download_artifacts.py                   # all benchmarks
    python scripts/download_artifacts.py --benchmark e2       # one benchmark (prefix match)
    python scripts/download_artifacts.py --list            # show registry + local status
    python scripts/download_artifacts.py --force           # re-download even if present
"""
from __future__ import annotations

import argparse
import sys

from pal.artifacts.core import (
    HF_REPO_ID,
    download_artifact,
    get_data_dir,
)
from pal.artifacts.registry import REGISTRY


def _select(benchmark: str | None) -> list[tuple[str, list]]:
    if benchmark is None:
        return list(REGISTRY.items())
    matches = [(k, v) for k, v in REGISTRY.items() if k.startswith(benchmark)]
    if not matches:
        known = ", ".join(sorted(REGISTRY)) or "(empty)"
        sys.exit(f"no benchmarks match {benchmark!r}. registered: {known}")
    return matches


def _cmd_list() -> None:
    data_dir = get_data_dir()
    print(f"HF repo: {HF_REPO_ID}")
    print(f"PAL_DATA_DIR: {data_dir}")
    if not REGISTRY:
        print("(registry is empty)")
        return
    for bench_id, refs in REGISTRY.items():
        print(f"\n{bench_id}")
        for ref in refs:
            if ref.location == "in_repo":
                print(f"  [in-repo] {ref.name} -> {ref.repo_path}")
                continue
            target = data_dir / bench_id / ref.local_subdir / ref.hf_filename.rsplit("/", 1)[-1]
            status = "present" if target.exists() else "missing"
            print(f"  [{status}] {ref.hf_filename} @ {ref.hf_revision[:8]}")


def _cmd_download(benchmark: str | None, force: bool) -> None:
    for bench_id, refs in _select(benchmark):
        print(f"\n=== {bench_id} ===")
        for ref in refs:
            if ref.location == "in_repo":
                print(f"  skip (in-repo): {ref.name} -> {ref.repo_path}")
                continue
            target_dir = get_data_dir() / bench_id / ref.local_subdir
            target = target_dir / ref.hf_filename.rsplit("/", 1)[-1]
            if target.exists() and not force:
                print(f"  skip (present): {ref.hf_filename}")
                continue
            print(f"  download: {ref.hf_filename} @ {ref.hf_revision[:8]}")
            path = download_artifact(bench_id, ref)
            print(f"    -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", help="benchmark id prefix (e.g. e2)")
    p.add_argument("--list", action="store_true", help="list registry + local status and exit")
    p.add_argument("--force", action="store_true", help="re-download already-present files")
    args = p.parse_args()

    if args.list:
        _cmd_list()
        return
    _cmd_download(args.benchmark, args.force)


if __name__ == "__main__":
    main()
