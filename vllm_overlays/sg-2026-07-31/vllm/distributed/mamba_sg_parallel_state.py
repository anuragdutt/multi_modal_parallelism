from enum import Enum

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


class MambaSGRole(str, Enum):
    GATE = "gate"
    SSM = "ssm"


def validate_mamba_sg_sizes(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> None:
    tp_world_size = get_tensor_model_parallel_world_size()

    if gate_parallel_size < 1:
        raise ValueError(
            f"gate_parallel_size must be positive, got {gate_parallel_size}"
        )

    if ssm_parallel_size < 1:
        raise ValueError(
            f"ssm_parallel_size must be positive, got {ssm_parallel_size}"
        )

    if gate_parallel_size + ssm_parallel_size != tp_world_size:
        raise ValueError(
            "Mamba SG rank counts must cover exactly one tensor-parallel "
            "group. "
            f"Got gate={gate_parallel_size}, "
            f"ssm={ssm_parallel_size}, "
            f"tp={tp_world_size}."
        )

    if gate_parallel_size != ssm_parallel_size:
        raise ValueError("Mamba SG currently requires equal gate and SSM groups")


def get_mamba_sg_role(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> MambaSGRole:
    validate_mamba_sg_sizes(
        gate_parallel_size=gate_parallel_size,
        ssm_parallel_size=ssm_parallel_size,
    )

    tp_rank = get_tensor_model_parallel_rank()

    if tp_rank < gate_parallel_size:
        return MambaSGRole.GATE

    return MambaSGRole.SSM

def is_gate_model_rank(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> bool:
    return (
        get_mamba_sg_role(
            gate_parallel_size,
            ssm_parallel_size,
        )
        == MambaSGRole.GATE
    )


def is_ssm_model_rank(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> bool:
    return (
        get_mamba_sg_role(
            gate_parallel_size,
            ssm_parallel_size,
        )
        == MambaSGRole.SSM
    )


def get_gate_model_parallel_rank(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> int:
    tp_rank = get_tensor_model_parallel_rank()

    if not is_gate_model_rank(
        gate_parallel_size,
        ssm_parallel_size,
    ):
        raise RuntimeError(
            f"TP rank {tp_rank} is an SSM rank, not a Gate rank."
        )

    # Gate occupies the first TP ranks.
    return tp_rank


def get_ssm_model_parallel_rank(
    gate_parallel_size: int,
    ssm_parallel_size: int,
) -> int:
    tp_rank = get_tensor_model_parallel_rank()

    if not is_ssm_model_rank(
        gate_parallel_size,
        ssm_parallel_size,
    ):
        raise RuntimeError(
            f"TP rank {tp_rank} is a Gate rank, not an SSM rank."
        )

    # SSM rank numbering starts again from zero.
    return tp_rank - gate_parallel_size