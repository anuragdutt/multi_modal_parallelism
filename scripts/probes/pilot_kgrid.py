"""Print graph step time versus split point k from a pilot sweep CSV."""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/raw/sweep_pilot.csv")
g = df[(df.variant == "graph") & (df.op == "step_max")].groupby(["ctx", "S", "cuts_src", "cuts"], dropna=False)["median_ms"].median().reset_index()
for c in sorted(g.ctx.unique()):
    print(f"--- c={c} graph step_max (ms) vs k")
    for S in sorted(g.S.unique()):
        s = g[(g.ctx == c) & (g.S == S)].sort_values("cuts")
        if S == 0:
            continue
        print(f"  S={S}: " + ", ".join(f"{src}:{int(float(k)) if pd.notna(k) and str(k) != '' else '-'}={v:.3f}" for src, k, v in zip(s.cuts_src, s.cuts, s.median_ms)))
    base = df[(df.variant == "graph") & (df.op == "step_max") & (df.ctx == c) & (df.S == 0.0)].groupby("mode")["median_ms"].median()
    print("  base modes: " + ", ".join(f"{m}={v:.3f}" for m, v in base.items()))
