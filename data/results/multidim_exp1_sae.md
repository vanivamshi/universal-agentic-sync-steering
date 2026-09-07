# Experiment 1 — broad GAP L4 SAE (Engels lens)

train_rows=2223 features=4096 dead=0.648
transcripts=54 surface_pos=8 control=24

## Decision: `SAE_HIT`
≥1 irreducible co-occurring feature cluster with |Cohen d|≥0.50 vs control.

Clusters (size 2–12): 13

### Cluster 0: feats=[1159, 4073, 1674, 2477] size=4 cooc=0.411 irr=True
- Cohen d surface vs control=3.752 vs nongap=-0.25204846182294083 hit=True

### Cluster 1: feats=[1736, 3893, 2361, 2916, 3724, 698, 1140, 3260, 2216, 2058, 2639, 2728] size=12 cooc=0.498 irr=True
- Cohen d surface vs control=3.166 vs nongap=-0.3821382046976676 hit=True

### Cluster 2: feats=[688, 3466, 153, 139, 2579] size=5 cooc=0.566 irr=True
- Cohen d surface vs control=2.931 vs nongap=0.03168279138669 hit=True

### Cluster 3: feats=[2139, 521] size=2 cooc=0.677 irr=True
- Cohen d surface vs control=2.588 vs nongap=0.11814923464108136 hit=True

### Cluster 4: feats=[4055, 2824] size=2 cooc=0.588 irr=True
- Cohen d surface vs control=-2.381 vs nongap=-0.6645497896087394 hit=True

### Cluster 5: feats=[3040, 764] size=2 cooc=0.588 irr=True
- Cohen d surface vs control=1.999 vs nongap=0.2729959116122422 hit=True

### Cluster 6: feats=[733, 1015, 3331, 969, 3861, 133, 1426, 309, 3505] size=9 cooc=0.347 irr=False
- Cohen d surface vs control=1.633 vs nongap=0.7882877363292884 hit=False

### Cluster 7: feats=[3866, 464, 147, 1361] size=4 cooc=0.316 irr=False
- Cohen d surface vs control=-0.453 vs nongap=-0.4361788728300886 hit=False

### Cluster 8: feats=[830, 2395] size=2 cooc=0.500 irr=True
- Cohen d surface vs control=-0.403 vs nongap=-1.1321235563945151 hit=False

### Cluster 9: feats=[925, 3937, 3314, 2182, 3762, 3568, 3558] size=7 cooc=0.341 irr=False
- Cohen d surface vs control=0.357 vs nongap=0.2630862368008077 hit=False

### Cluster 10: feats=[78, 2557, 1279] size=3 cooc=0.301 irr=False
- Cohen d surface vs control=0.293 vs nongap=0.5153565458396778 hit=False

### Cluster 11: feats=[790, 2481] size=2 cooc=0.149 irr=False
- Cohen d surface vs control=-0.038 vs nongap=0.29975287158674613 hit=False

Exp3 licensed (descriptive): True
SAE weights: `/Users/vamshisunku.mohan/bluedot_alignment/data/directions/multidim_exp1_sae.pt`
Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/multidim_exp1_sae.json`

**Caveat:** Control pool = `real_gap` legitimate trajectories (not in
`gap_deception.json`). Large |Cohen d| vs control with near-zero d vs nongap
eliciting is consistent with **eliciting↔benign domain shift**, not a clean
`surface_gap`-specific unit. Exp3 still tested top clusters under locked J bars.
