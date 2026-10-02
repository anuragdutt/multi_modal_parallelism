"""Print speedup over tp4 for every mode in a per-model sweep CSV (min over repeats, graph variant)."""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
g = df[(df.variant == "graph") & (df.op == "step_max")].groupby(["mode", "batch", "ctx"])["median_ms"].min().unstack("mode")
base = g["tp4"]
cols = [c for c in g.columns if c != "tp4"]
out = pd.DataFrame({"tp4_ms": base.round(3), **{c: (base / g[c]).round(2) for c in cols}})
print(out.to_string())
