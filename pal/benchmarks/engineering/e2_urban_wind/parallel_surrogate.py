"""Multi-GPU batch-parallel wrapper around LTXSurrogate (e2 only).

Splits the input batch dim across N CUDA devices, runs replicas concurrently
in threads, gathers outputs back to cuda:0. Single-process (no DDP/NCCL).
"""

from __future__ import annotations

import copy
import threading
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ParallelSurrogate(nn.Module):
    """Batch-parallel wrapper around LTXSurrogate across multiple CUDA devices.

    Forward signature matches LTXSurrogate exactly:
      (building_mask, inlet_u, inlet_v, seed=42) -> (u, v)

    Replication: deepcopy the base surrogate once per extra GPU. All replicas
    share the same (frozen) weights but live on distinct devices. Memory cost
    on N GPUs ~ N * 12.7 GB (fp32 weights) + per-GPU activations.

    Concurrency: each replica runs in its own thread. The Python GIL releases
    during CUDA kernel launches, so the threads dispatch work concurrently to
    their respective GPUs. Backward is handled by PyTorch's autograd engine,
    which has per-device worker threads.
    """

    def __init__(self, base: nn.Module, devices: list[str]):
        """
        Args:
            base: An LTXSurrogate already loaded on devices[0].
            devices: List of CUDA device strings (e.g. ["cuda:0", "cuda:1"]).
                     Length determines parallelism. Must contain >= 1 device.
        Returns:
            None.
        """
        super().__init__()
        if len(devices) < 1:
            raise ValueError("ParallelSurrogate needs >=1 device")
        self.devices: list[str] = list(devices)

        # Move base to CPU before deepcopy so the copy is not first allocated on devices[0].
        base.cpu()
        self._move_unregistered_tensors(base, "cpu")
        replicas: list[nn.Module] = [base]
        for d in self.devices[1:]:
            r = copy.deepcopy(base).to(d)
            self._move_unregistered_tensors(r, d)
            r.eval()
            replicas.append(r)
        base.to(self.devices[0])
        self._move_unregistered_tensors(base, self.devices[0])

        # Re-bind the per-resnet checkpoint wrappers: their closures capture the base replica.
        for r in replicas:
            self._rebind_vae_decoder_checkpointing(r)

        self.replicas = nn.ModuleList(replicas)

    @classmethod
    def _move_unregistered_tensors(cls, module: nn.Module, device: str) -> None:
        """Move plain tensor attrs that `module.to(device)` does not touch.

        Some third-party modules stash device-bound tensors directly on Python
        attrs rather than registering them as parameters or buffers. Those
        tensors survive deepcopy on their original device, so cloned replicas
        can fail with cross-device errors during forward.
        """

        seen: set[int] = set()

        def _move_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                moved = [_move_value(item) for item in value]
                return moved if any(a is not b for a, b in zip(value, moved, strict=False)) else value
            if isinstance(value, tuple):
                moved = tuple(_move_value(item) for item in value)
                return moved if any(a is not b for a, b in zip(value, moved, strict=False)) else value
            if isinstance(value, dict):
                moved = {key: _move_value(item) for key, item in value.items()}
                return moved if any(moved[key] is not value[key] for key in value) else value
            return value

        def _visit(current: nn.Module) -> None:
            module_id = id(current)
            if module_id in seen:
                return
            seen.add(module_id)

            registered = (
                set(current._parameters)
                | set(current._buffers)
                | set(current._modules)
            )
            for name, value in vars(current).items():
                if name in registered:
                    continue
                moved = _move_value(value)
                if moved is not value:
                    setattr(current, name, moved)

            for child in current.children():
                _visit(child)

        _visit(module)

    @staticmethod
    def _rebind_vae_decoder_checkpointing(replica: nn.Module) -> None:
        """Reinstall per-resnet checkpoint wrappers on this replica.

        The vendor surrogate monkey-patches each decoder resnet with a closure
        that captures the original bound `forward` method. After deepcopy,
        those closures can still reference the base replica on cuda:0, so a
        cuda:1+ shard may execute against cuda:0-owned tensors and fail with
        cross-device errors.
        """

        decoder = replica.adapted_vae.vae.decoder

        def _wrap_resnet(resnet: nn.Module) -> None:
            orig_forward = type(resnet).forward

            def ckpt_forward(*args, **kwargs):
                return checkpoint(
                    lambda *inner_args, **inner_kwargs: orig_forward(
                        resnet, *inner_args, **inner_kwargs
                    ),
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )

            resnet.forward = ckpt_forward

        if hasattr(decoder, "mid_block") and hasattr(decoder.mid_block, "resnets"):
            for resnet in decoder.mid_block.resnets:
                _wrap_resnet(resnet)

        if hasattr(decoder, "up_blocks"):
            for block in decoder.up_blocks:
                if hasattr(block, "resnets"):
                    for resnet in block.resnets:
                        _wrap_resnet(resnet)

    @property
    def n_replicas(self) -> int:
        return len(self.replicas)

    def forward(
        self,
        building_mask: torch.Tensor,
        inlet_u: torch.Tensor,
        inlet_v: torch.Tensor,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter the batch across replicas, run in parallel, gather to cuda:0.

        Args:
            building_mask: (B, H, W) or (B, 1, H, W) on devices[0].
            inlet_u: (B,) on devices[0].
            inlet_v: (B,) on devices[0].
            seed: Surrogate noise seed (fixed for clean gradient signal).
        Returns:
            (u, v) tuple, each (B, T, H, W) on devices[0].
        """
        n = self.n_replicas
        B = building_mask.shape[0]

        if n == 1 or B == 1:
            return self.replicas[0](building_mask, inlet_u, inlet_v, seed=seed)

        bm_shards = torch.tensor_split(building_mask, n, dim=0)
        iu_shards = torch.tensor_split(inlet_u, n, dim=0)
        iv_shards = torch.tensor_split(inlet_v, n, dim=0)

        outputs: list[Any] = [None] * n
        errors: list[BaseException | None] = [None] * n

        def _run(i: int) -> None:
            try:
                torch.cuda.set_device(self.devices[i])
                bm = bm_shards[i].to(self.devices[i], non_blocking=True)
                iu = iu_shards[i].to(self.devices[i], non_blocking=True)
                iv = iv_shards[i].to(self.devices[i], non_blocking=True)
                u, v = self.replicas[i](bm, iu, iv, seed=seed)
                outputs[i] = (u, v)
            except BaseException as e:
                errors[i] = e

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(
                    f"ParallelSurrogate replica {i} ({self.devices[i]}) "
                    f"failed: {err!r}"
                ) from err

        u_parts = [outputs[i][0].to(self.devices[0]) for i in range(n)]
        v_parts = [outputs[i][1].to(self.devices[0]) for i in range(n)]
        return torch.cat(u_parts, dim=0), torch.cat(v_parts, dim=0)


def detect_cuda_devices() -> list[str]:
    """Return a list of all visible CUDA devices, or [] if none.

    Returns:
        List of device strings like ["cuda:0", "cuda:1", ...]. Empty if no CUDA.
    """
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


def memory_warning(per_gpu_batch: int, est_gb_per_sample: float = 3.0) -> str | None:
    """Build a memory warning string if requested per-GPU batch looks risky.

    Uses an fp32 estimate of ~1.5-3 GB/sample on top of the 12.7 GB frozen
    surrogate weights.

    Args:
        per_gpu_batch: Samples that will land on each GPU.
        est_gb_per_sample: Per-sample activation budget (fp32). Default 3.0.
    Returns:
        Warning string, or None if memory looks fine.
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024**3
    needed_gb = per_gpu_batch * est_gb_per_sample
    if needed_gb > free_gb * 0.9:
        return (
            f"WARNING: per-GPU batch={per_gpu_batch} estimated to need "
            f"~{needed_gb:.1f} GB activations; only ~{free_gb:.1f} GB free. "
            f"Likely CUDA OOM. Lower --batch-size or request more GPUs."
        )
    return None
