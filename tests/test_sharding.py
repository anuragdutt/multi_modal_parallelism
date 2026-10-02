import pytest

from harness.config import FalconH1Dims
from harness.sharding import (
    attn_shard, build_mup_vector, k_grid, kv_replication_map, mlp_shard, plan_for, ssm_shard, swing_width,
)

D7 = FalconH1Dims(
    hidden=3072, intermediate=12288, n_q=12, n_kv=2, head_dim=128, n_mamba_heads=24, mamba_head_dim=128,
    d_ssm=3072, d_state=256, n_groups=1, d_conv=4, chunk=256, rms_eps=1e-5, rope_theta=1e11, max_pos=262144,
    attn_in=1.0, attn_out=0.1041666, key_mult=0.0307, ssm_in=0.4166666, ssm_out=0.1178511,
    ssm_mults=(0.3535533, 0.25, 0.1767766, 0.5, 0.3535533), mlp_mults=(0.2946278, 0.032552),
)
D34 = FalconH1Dims(
    hidden=5120, intermediate=21504, n_q=20, n_kv=4, head_dim=128, n_mamba_heads=32, mamba_head_dim=128,
    d_ssm=4096, d_state=256, n_groups=2, d_conv=4, chunk=128, rms_eps=1e-5, rope_theta=1e11, max_pos=262144,
    attn_in=1.0, attn_out=0.0375, key_mult=0.011, ssm_in=0.25, ssm_out=0.0884,
    ssm_mults=(0.3535533, 0.25, 0.1767766, 0.5, 0.3535533), mlp_mults=(0.1767766, 0.0112),
)


def _covers(ranges, total):
    seen = sorted(i for r in ranges for i in r)
    assert seen == list(range(total)), "ranges do not partition exactly once"


def test_tp4_attention_partition_and_replication():
    shards = [attn_shard(D7, r, 4) for r in range(4)]
    _covers([s.q_heads for s in shards], 12)
    assert kv_replication_map(D7, 4) == {0: (0, 1), 1: (2, 3)}
    assert all(s.kv_replicas == 2 for s in shards)
    assert shards[1].q_rows == slice(384, 768) and shards[1].k_rows == slice(0, 128)
    assert shards[3].o_cols == slice(1152, 1536)


def test_uneven_attention_shards_3b():
    D3 = FalconH1Dims(**{**D7.__dict__, "hidden": 2560, "intermediate": 6144, "n_q": 10, "n_mamba_heads": 32, "d_ssm": 4096})
    shards = [attn_shard(D3, r, 4) for r in range(4)]
    _covers([s.q_heads for s in shards], 10)
    assert [len(s.q_heads) for s in shards] == [3, 2, 3, 2]
    assert kv_replication_map(D3, 4) == {0: (0, 1), 1: (2, 3)}
    assert [attn_shard(D3, r, 2).q_heads for r in range(2)] == [range(0, 5), range(5, 10)]


def test_split22_attention_no_replication():
    shards = [attn_shard(D7, r, 2) for r in range(2)]
    _covers([s.q_heads for s in shards], 12)
    assert kv_replication_map(D7, 2) == {0: (0,), 1: (1,)}
    assert shards[1].q_rows == slice(768, 1536) and shards[1].k_rows == slice(128, 256)


def test_tp4_ssm_partition_and_replicated_groups():
    shards = [ssm_shard(D7, r, 4, (0, 1, 2, 3)) for r in range(4)]
    _covers([s.heads for s in shards], 24)
    for s in shards:
        assert s.b_rows == slice(6144, 6400) and s.c_rows == slice(6400, 6656)
        assert s.norm_ranks == (0, 1, 2, 3)
        assert s.local_in_rows == (768, 768, 256, 256, 6)
    assert shards[2].z_rows == slice(1536, 2304) and shards[2].x_rows == slice(3072 + 1536, 3072 + 2304)
    assert shards[2].dt_rows == slice(6656 + 12, 6656 + 18)
    assert shards[2].conv_rows == (slice(1536, 2304), slice(3072, 3328), slice(3328, 3584))


def test_split22_ssm_pair_norm():
    shards = [ssm_shard(D7, s, 2, (2, 3)) for s in range(2)]
    _covers([s.heads for s in shards], 24)
    assert shards[0].norm_ranks == (2, 3) and shards[1].norm_ranks == (2, 3)
    assert shards[1].local_in_rows == (1536, 1536, 256, 256, 12)
    assert shards[1].out_cols == slice(1536, 3072)


def test_34b_tp4_group_aware():
    shards = [ssm_shard(D34, r, 4, (0, 1, 2, 3)) for r in range(4)]
    _covers([s.heads for s in shards], 32)
    assert [list(s.groups) for s in shards] == [[0], [0], [1], [1]]
    assert shards[0].norm_ranks == (0, 1) and shards[3].norm_ranks == (2, 3)
    assert shards[2].b_rows == slice(2 * 4096 + 256, 2 * 4096 + 512)
    assert shards[2].c_rows == slice(2 * 4096 + 2 * 256 + 256, 2 * 4096 + 2 * 256 + 512)
    a = [attn_shard(D34, r, 4) for r in range(4)]
    assert kv_replication_map(D34, 4) == {0: (0,), 1: (1,), 2: (2,), 3: (3,)}
    assert a[0].q_heads == range(0, 5)


def test_mup_vector_matches_vllm_formula():
    for G in (1, 2, 4):
        s = ssm_shard(D7, 0, G, tuple(range(G)))
        v = build_mup_vector(D7, s)
        d = 3072 // G
        assert v.shape == (1, 2 * d + 2 * 256 + 24 // G)
        assert float(v[0, 0]) == pytest.approx(0.3535533) and float(v[0, d]) == pytest.approx(0.25)
        assert float(v[0, 2 * d]) == pytest.approx(0.1767766) and float(v[0, 2 * d + 256]) == pytest.approx(0.5)
        assert float(v[0, -1]) == pytest.approx(0.3535533)


def test_swing_bookkeeping():
    assert swing_width(0.1, 3072) == 320 and swing_width(0.4, 3072) == 1216 and swing_width(1.0, 3072) == 3072
    w = 640
    # pair (0 attn, 2 ssm); neutral k = w reproduces plain TP exactly
    a = mlp_shard(D7, 0, 4, 0.2, None, "attn", 2)
    s = mlp_shard(D7, 2, 4, 0.2, None, "ssm", 0)
    assert a.w == w and a.k == w and s.k == w
    assert len(a.col_order) == 3072 - w + 2 * w == len(s.col_order)
    assert sorted(a.active_global_cols()) == list(range(0, 3072))
    assert sorted(s.active_global_cols()) == list(range(2 * 3072, 3 * 3072))
    # any k partitions the pair's columns exactly once, and active rows are prefixes of the pool
    pair_cols = set(range(0, 3072)) | set(range(2 * 3072, 3 * 3072))
    for k in k_grid(w, 128):
        a = mlp_shard(D7, 0, 4, 0.2, k, "attn", 2)
        s = mlp_shard(D7, 2, 4, 0.2, k, "ssm", 0)
        assert a.active_rows + s.active_rows == 2 * 3072
        got = list(a.active_global_cols()) + list(s.active_global_cols())
        assert len(got) == len(set(got)) and set(got) == pair_cols
    assert k_grid(w, 128)[-1] == 2 * w
    # tp modes have no pool
    t = mlp_shard(D7, 1, 4, 0.0, None)
    assert t.w == 0 and t.active_rows == 3072 and list(t.col_order) == list(range(3072, 6144))


def test_plan_modes():
    p = plan_for("tp1", 0, 1, D7)
    assert p.combine == "none" and p.ssm.norm_ranks == (0,)
    for r in range(4):
        p = plan_for("tp4", r, 4, D7)
        assert p.combine == "allreduce_each" and p.has_attn and p.has_ssm
        p = plan_for("split22", r, 4, D7, swing=0.3)
        assert p.combine == "sum_then_allreduce"
        assert p.has_attn == (r < 2) and p.has_ssm == (r >= 2)
        assert p.mlp.role == ("attn" if r < 2 else "ssm") and p.mlp.partner == (r + 2 if r < 2 else r - 2)
    with pytest.raises(ValueError):
        plan_for("tp4", 0, 4, D7, swing=0.2)
