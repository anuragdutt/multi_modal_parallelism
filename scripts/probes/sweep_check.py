import pandas as pd, numpy as np
df = pd.read_csv("results/raw/sweep_full.csv")
g = df[(df.variant=="graph")&(df.op=="step_max")]
print("=== B=64 cells: per-repeat graph step_max (ms), base modes ===")
for c in (8192, 32768):
    for m in ("tp4","tp4_fused","tp4_streams","split22"):
        s = g[(g.batch==64)&(g.ctx==c)&(g["mode"]==m)&(g.S==0.0)].sort_values("repeat")
        print(f"  c={c} {m:12s}: " + ", ".join(f"rep{int(r.repeat)}={r.median_ms:.3f} (p10 {r.p10_ms:.3f}, p90 {r.p90_ms:.3f}, {r.power_w:.0f}W {r.util_pct}%)" for r in s.itertuples()))
print("=== CoV across repeats of graph step_max, worst 8 cells ===")
cov = g.groupby(["mode","S","cuts_src","batch","ctx"])["median_ms"].agg(lambda x: x.std()/x.mean()).sort_values(ascending=False)
print(cov.head(8).to_string())
print("=== contamination check: rows with util>0 or power>120W at measurement start ===")
e = df[(df.op=="step")&(df["rank"]>=0)]
print(f"  rows: {len(e)}, power>120W: {(e.power_w>120).sum()}, util>0: {(e.util_pct>0).sum()}; power range {e.power_w.min():.0f}-{e.power_w.max():.0f} W")
print("=== byte model: predicted vs measured (graph), per step and per op ===")
v = df[(df.variant=="graph")&(df["rank"]>=0)&(df.op=="step")&(df.ms_pred.notna())].copy()
v["ms_pred"]=pd.to_numeric(v.ms_pred, errors="coerce"); v=v[v.ms_pred>0]
err = ((v.median_ms - v.ms_pred).abs()/v.ms_pred)
print(f"  per-step (graph): n={len(v)} median rel err {err.median():.1%}, p90 {err.quantile(0.9):.1%}; measured/pred median ratio {(v.median_ms/v.ms_pred).median():.2f}")
for m in ("tp4","split22"):
    vv = v[v["mode"]==m]; ee=((vv.median_ms - vv.ms_pred).abs()/vv.ms_pred)
    print(f"    {m}: median rel err {ee.median():.1%}, ratio {(vv.median_ms/vv.ms_pred).median():.2f}")
ve = df[(df.variant=="eager")&(df["rank"]>=0)&(df.op!="step")&(df.op!="step_wall")&(df.ms_pred.notna())].copy()
ve["ms_pred"]=pd.to_numeric(ve.ms_pred, errors="coerce"); ve=ve[ve.ms_pred>0]
for op in ("attn.fa","ssm.ssu","mlp.gateup","ssm.inproj"):
    vv=ve[ve.op==op]; ee=((vv.median_ms - vv.ms_pred).abs()/vv.ms_pred)
    print(f"  eager op {op}: median rel err {ee.median():.1%}, measured/pred {(vv.median_ms/vv.ms_pred).median():.2f}")
