"""Print Nemotron 3 Nano layer tensor names/shapes for layers 0 (Mamba), 1 (MoE), 5 (attention) and config fields."""
import json, os, sys
from safetensors import safe_open
s = sys.argv[1]
idx = json.load(open(os.path.join(s, "model.safetensors.index.json")))["weight_map"]
for L in (0, 1, 5):
    keys = sorted(k for k in idx if k.startswith(f"backbone.layers.{L}."))
    shards = sorted(set(idx[k] for k in keys))
    print(f"--- layer {L}: {len(keys)} tensors; shards {shards}")
    shown = [k for k in keys if ".experts." not in k or ".experts.0." in k]
    for k in shown[:40]:
        with safe_open(os.path.join(s, idx[k]), "pt") as f:
            sl = f.get_slice(k)
            print(f"  {k} {sl.get_shape()} {sl.get_dtype()}")
cfg = json.load(open(os.path.join(s, "config.json")))
keys = ["hidden_size", "num_hidden_layers", "hybrid_override_pattern", "num_attention_heads", "num_key_value_heads", "head_dim",
        "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel", "n_routed_experts", "num_experts_per_tok",
        "moe_intermediate_size", "moe_shared_expert_intermediate_size", "n_shared_experts", "mlp_hidden_act", "rms_norm_eps",
        "rope_theta", "attention_bias", "mamba_hidden_act", "mamba_proj_bias", "mamba_conv_bias", "router_bias", "norm_topk_prob",
        "moe_router_scaling", "rotary_percentage", "partial_rotary_factor", "intermediate_size", "expand", "chunk_size", "layer_norm_epsilon", "use_bias"]
print({k: cfg.get(k) for k in keys})
