# Factorized 8-way controller (Qwen)

> **CAVEAT — exploratory.** Free-run corpus had H=1 in only 1/16 episodes, so V was not
> Stage-2 validated. Baseline often S=[1,0,1] (check did not trigger private load).
> This run uses unfrozen candidate V via `--allow-unfrozen` (activation path live).
> Not a claim that all 8 targets are controllable.


- Reached S=m*: **2/8**
- Median ΔE: 0.50

| m* | S_before | S_after | E_before | E_after | ΔE | reached |
|----|----------|---------|----------|---------|----|---------|
| (0, 0, 0) | [1, 0, 1] | [1, 0, 0] | 2 | 1 | 1 | False |
| (0, 0, 1) | [1, 0, 1] | [1, 0, 0] | 1 | 2 | -1 | False |
| (0, 1, 0) | [1, 0, 1] | [1, 0, 0] | 3 | 2 | 1 | False |
| (0, 1, 1) | [1, 0, 1] | [1, 0, 0] | 2 | 3 | -1 | False |
| (1, 0, 0) | [1, 0, 1] | [1, 0, 0] | 1 | 0 | 1 | True |
| (1, 0, 1) | [1, 0, 1] | [1, 0, 1] | 0 | 0 | 0 | True |
| (1, 1, 0) | [1, 0, 1] | [1, 0, 0] | 2 | 1 | 1 | False |
| (1, 1, 1) | [1, 0, 1] | [1, 0, 0] | 1 | 2 | -1 | False |
