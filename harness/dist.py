"""vLLM distributed groups for the harness: the tensor-parallel group over all ranks plus a two-rank
"branch pair" group ([0,1],[2,3]) used by the split modes. Both are created on every rank in every mode so
communicator setup is identical across modes."""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist


@dataclass
class Groups:
    rank: int
    local_rank: int
    world_size: int
    tp: object  # vllm GroupCoordinator
    pair: object | None  # vllm GroupCoordinator or None (world_size != 4)
    device: torch.device

    def group_for_ranks(self, ranks: tuple[int, ...]):
        """Return the coordinator whose membership equals `ranks`, or None for a single rank."""
        if len(ranks) <= 1:
            return None
        if len(ranks) == self.world_size:
            return self.tp
        if self.pair is not None and len(ranks) == 2:
            return self.pair
        raise ValueError(f"no communicator for ranks {ranks}")


_CONFIG_STACK = None


def ensure_vllm_config(world: int) -> None:
    """vLLM >= 0.30 reads the current engine config inside initialize_model_parallel and in CustomOps.
    Enter a default config for the lifetime of the process (what vLLM's own tests do via a fixture)."""
    global _CONFIG_STACK
    if _CONFIG_STACK is not None:
        return
    from contextlib import ExitStack

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    cfg = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    _CONFIG_STACK = ExitStack()
    _CONFIG_STACK.enter_context(set_current_vllm_config(cfg))


def init_groups() -> Groups:
    from vllm.distributed import parallel_state as ps

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    ensure_vllm_config(world)
    ps.init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://",
                                    local_rank=local_rank, backend="nccl")
    ps.initialize_model_parallel(tensor_model_parallel_size=world)
    tp = ps.get_tp_group()
    pair = None
    if world == 4:
        pair = ps.init_model_parallel_group([[0, 1], [2, 3]], local_rank, "nccl", group_name="branch_pair")
    return Groups(rank=rank, local_rank=local_rank, world_size=world, tp=tp, pair=pair, device=device)


def describe_comm(g) -> dict:
    if g is None:
        return {"ranks": None}
    info = {"ranks": list(getattr(g, "ranks", [])), "world_size": getattr(g, "world_size", None)}
    dc = getattr(g, "device_communicator", None)
    ca = getattr(dc, "ca_comm", None) if dc is not None else None
    py = getattr(dc, "pynccl_comm", None) if dc is not None else None
    info["custom_allreduce"] = bool(ca is not None and not getattr(ca, "disabled", True))
    info["custom_allreduce_max_size"] = getattr(ca, "max_size", None) if ca is not None else None
    info["pynccl"] = bool(py is not None and not getattr(py, "disabled", False))
    try:
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary  # noqa: F401

        info["nccl_version"] = getattr(py, "nccl_version", None) if py is not None else None
    except Exception:  # pragma: no cover
        info["nccl_version"] = None
    return info


@contextmanager
def capture_context(groups: Groups) -> Iterator[object]:
    """Enter vLLM's graph-capture contexts for the TP group (module-level) and the pair group."""
    from vllm.distributed import parallel_state as ps

    with ps.graph_capture(groups.device) as ctx:
        if groups.pair is not None:
            with groups.pair.graph_capture(ctx):
                yield ctx
        else:
            yield ctx


def barrier_sync() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def destroy() -> None:
    from vllm.distributed import parallel_state as ps

    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _selftest() -> None:
    import json

    g = init_groups()
    x = torch.ones(64, 3072, device=g.device, dtype=torch.bfloat16) * (g.rank + 1)
    y = g.tp.all_reduce(x)
    expected = sum(range(1, g.world_size + 1))
    ok_tp = bool(torch.all(y == expected))
    ok_pair = None
    if g.pair is not None:
        z = torch.ones(64, 1, device=g.device, dtype=torch.float32) * (g.rank + 1)
        zz = g.pair.all_reduce(z)
        exp_pair = (1 + 2) if g.rank < 2 else (3 + 4)
        ok_pair = bool(torch.all(zz == exp_pair))
    barrier_sync()
    print(json.dumps({"rank": g.rank, "tp_ok": ok_tp, "pair_ok": ok_pair, "tp": describe_comm(g.tp), "pair": describe_comm(g.pair)}))
    destroy()


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
