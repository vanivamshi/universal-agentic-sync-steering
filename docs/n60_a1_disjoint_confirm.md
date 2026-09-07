# n60 mean_A1 false→true: disjoint confirm

The 3+3 `NULL` and 10+10 `SATURATE` runs are **nested on the same
sorted files** (`_pick_domain` takes the first *n*). Files 0–5 ⊂ files 0–19.
That is expansion, not a second corpus. Do not treat it as independent
replication. Do not emit a steer vector. SATURATE still selected `[]`.

**Confirm set (locked):** coding + metacognitive files **after index 20**
(the 40 files never used in either pick). Same k=8 residual mean, same
Cohen *d*, same p90-of-|d_random| rule, seed `20260814`, 40 random dirs.
No ε*. No gate retune.

| Test | What | Pass |
|---|---|---|
| T0 nested | first 6 filenames ⊂ first 20 | diagnostic only |
| T1 transfer | freeze mean_A1 from n20 **rank** (files 0–9); score leftover files 20–39 | \|d\| > leftover random p90 |
| T2 replica | new mean_A1 from leftover rank (20–29); score leftover hold (30–39) | \|d\| > leftover-hold random p90 |

**`CONFIRM_HIT`** iff T1 and T2 both pass. **`CONFIRM_FAIL`** if either misses.
**`NESTED_ONLY`** if we never run leftover (this file exists so that case is named).

A T1/T2 hit still does **not** select a direction. It only says the |d| bar
flip survives files the first two runs never touched.
