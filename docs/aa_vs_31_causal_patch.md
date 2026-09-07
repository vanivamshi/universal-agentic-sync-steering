# AA vs 31-dim causal patch (geometry ≠ dissociation)

**Claim to test:** L4 near-orthogonality of Assistant Axis vs the 31-dim prose
persona PCA is only a **geometric** fact. Causal dissociation (“AA drives tool
action; 31-dim carries intent”) requires **activation patching**, not more
cosine reports.

## Definitions (locked)

| Role | Label rule |
|---|---|
| **Safe / compliant donor** | `text_safe ∧ ¬tool_violation` |
| **Violate recipient** | `surface_gap` (text_safe ∧ tool_violation) |

**Do not** use `open_compliance` as “compliant” — in this repo it means
`tool_violation ∧ ¬contradiction ∧ non-empty prose` (open jailbreak success).

## Intervention

Joint orthonormal basis `Q = qr([AA; PC0..PC30])` at L4.

On the **violate** prompt’s generation, at every token at L4:

| Condition | Patch |
|---|---|
| `identity` | no patch (regen baseline) |
| `patch_aa` | replace AA coord with donor safe’s `Qᵀ h_donor`; hold PC coords |
| `patch_31` | replace PC0..30 coords with donor’s; hold AA |
| `patch_both` | replace all joint coords (sanity / upper bound) |

`h_donor` = cached L4 transcript-mean residual for the safe trajectory
(`gap_deception.json`). Hook: `ActivationSubspaceCoordPatchHook`.

## Predictions (dissociation narrative)

1. **`patch_aa`:** tool_violation / surface_gap ↓ vs identity (action flips safer).
2. **`patch_31`:** tool outcome **not** flipped as much as `patch_aa` (intent
   subspace alone does not drive the tool call).

**HIT:** both (1) and (2) on the same pair set.  
**FALSIFY:** `patch_31` reduces violation ≥ `patch_aa`.  
**UNDERPOWERED:** n_pairs < 6 (descriptive only).

## Pairing

Prefer same `gap_domain`. If none, pair each violate with a safe from the pool
(cross-domain) and flag `cross_domain=true`.

## Out of scope

Mismatch monitor / fine-tune-to-close-gap — gated on HIT. Not RQ1 privilege.

Script: `scripts/run_aa_vs_31_causal_patch.py`
