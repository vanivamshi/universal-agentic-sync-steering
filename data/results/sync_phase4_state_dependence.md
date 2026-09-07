# Phase 4 diagnostic — state dependence of \(g_k\) (FROZEN)

> Pairwise compositionality ≠ sequential behavioral sync.
> High cross-cosine ≠ “no state dependence.”

$$
\boxed{
\text{state dependence}
=
\text{gradient orientation drift}
+
\text{gradient/margin magnitude change}
}
$$

α=1.5. Cross = j≠k.

- mean cos\((g_k(h_0),g_k(h_1))\) cross: **0.885**
- mean ‖Δg‖₂ cross: **0.432**
- geometry drifts under cross-steer: **True**

## Per target (after other-channel steer)

| k | orientation (mean cos) | min cos | ‖Δg‖ | magnitude (\|ΔM\|) |
|---|------------------------|---------|------|---------------------|
| C | 0.935 | 0.899 | 0.350 | 0.528 |
| **H** | **0.941** (stable) | 0.936 | 0.342 | **2.206** (large) |
| **O** | **0.779** (rotates) | **0.578** | **0.604** | 0.262 |

H: orientation stable, margin geometry changes a lot.
O: larger gradient rotation, smaller margin movement.

## Implication → Phase 5

Static \(v_c^{(k)}(h_0)\) is not trajectory-valid.
Next: adaptive \(v_c^{(k)}(h_t)\propto\operatorname{Proj}_{g_k(h_t)}(v_p^{(k)})\)
vs frozen \(v_c^{(k)}(h_0)\).

**8-way CLOSED.**
