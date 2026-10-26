# SG (SSM/gate split) scripts

- `smoke_sg.py`: in-container smoke and regression test (rank KV-cache groups + reorder threshold, waves determinism,
  SG vs TP first-token agreement). See the docstring for the three commands.
- E2 close-out (fix validation and the clean AIPerf re-measurement) is specified in `docs/PLAN.md` section 7; its
  scripts land here when it runs.
