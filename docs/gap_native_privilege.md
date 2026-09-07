# GAP-native v — RQ1 privilege (exploratory)

Native L4 tool-window mean-diff is a **dir-null HIT** and a **Tan LOW_AGREEMENT**
extract. p90 is not retuned. This run does **not** steer.

**Question:** relative to locked controls, does this v keep (or gain) geometric
privilege in `tool_call` windows vs prose? That is RQ1/RQ3 on the only
GAP-native candidate, without claiming the HIT licenses control.

| Item | Lock |
|---|---|
| v | `data/directions/gap_native_tool_sg_L4.jsonl` |
| controls | `controls_l4.jsonl` (syntax + domain_content) |
| random | first 8 of `random_controls_l4.jsonl` (frozen; not confirmatory n≥64) |
| perturb | k=4 → L=22, τ=0.5, relative blowup (RQ1 pilot recipe) |
| data | `gap_agentic.jsonl` (24) |
| Δ | ε*_tool − ε*_prose; **positive ⇒ privilege gain in tool** |

Primary contrast: I = Δ_native − mean Δ_learned.
RQ3-style: native extracted in tool mode should have **I > 0** (gains more
privilege in tool than learned prose controls). Null: I inside random Δ spread.

Script: `scripts/run_gap_native_privilege.py`
