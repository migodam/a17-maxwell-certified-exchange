# Review the verified-exchange efficiency follow-up

This directory adds deployment evidence to the completed Maxwell mechanism and conditional-design study. Start with [START_HERE](START_HERE.md), then [the final efficiency report](A17_EFFICIENT_VERIFICATION_FINAL.md). The [final scientific manuscript and its holdout/noise evidence](../closure_final/START_HERE.md) remain the paper baseline.

## Result and recommendation

**CONDITIONAL:** retain only the highest-ranked finalist in each exchange round, permit up to three accepted exchanges, and cache only material-independent current-bank geometry. Every accepted move still uses the original online numerical tolerance and a full directional `Jd` check at the current material state.

| Median measurement | Smooth voxel | Shell Gaussian |
|---|---:|---:|
| Original exchange / s | 581.346 | 592.709 |
| Recommended exchange / s | 425.031 | 535.598 |
| Reduction relative to original exchange | 26.89% | 9.64% |
| Recommended / unchanged receiver time | 1.958 | 1.318 |
| Full verification directions: original → recommended | 100 → 50 | 106 → 52 |
| Final complex-material error: original → recommended | 0.269856 → 0.263687 | 0.565989 → 0.541335 |
| Held-out measurement error: original → recommended | 0.00283650 → 0.00254582 | 0.01217928 → 0.01160040 |

The measurements cover two prescribed objects, current dimension k=8, and eighteen constrained nonlinear updates. Three timing repetitions estimate time variability, not independent physical samples. The supplementary top-1/cap-3 branch was admitted after the cap-2 quality failure, so it is not new independent holdout validation.

## What changed the engineering conclusion

Full finalist `Jd` used only 4.330/7.117 seconds of the original trajectories, about 1.19%/3.82% of the extra exchange time. The existing backend already shared complex128 CUDA LU and performed actual multiple-right-hand-side solves. Repeated anchor, candidate-bank and workspace construction was the larger directly measured increment. Exact geometry caching alone reduced full trajectory time by about 9%, with 1,308 paired array comparisons bitwise equal.

## Negative results to retain

- Top-1/cap-2 retained about 99% of frozen GN-gap improvement, yet worsened smooth-object final material error by 5.95% and held-out error by 21.14%.
- Exact upper-bound screening would need the full gradient at the changed step; the already-paid outer gradient is at step zero. It was not a free pruning route.
- Saved displacements were mostly full rank; previous-space reuse did not preserve quality; fixed empirical-radius triage did not save nonlinear verification.
- The recommended policy does not meet both original targets of at most two verification directions per update and exchange/receiver time at most 1.3. It improves the exchange implementation, while still costing more than receiver-only reconstruction.

## Evidence to inspect

1. [Profile](A17_EFFICIENCY_PROFILE.md): exclusive versus nested timing and unassigned time.
2. [Pareto curves](A17_EFFICIENCY_PARETO.md): frozen fidelity versus nonlinear quality.
3. [Claim ledger](A17_EFFICIENCY_CLAIM_LEDGER.md) and [failures](A17_EFFICIENCY_FAILURES_AND_LIMITS.md).
4. [Coverage](A17_EFFICIENCY_COVERAGE_FINAL.md), [cost](A17_EFFICIENCY_COST_FINAL.md), and [reproduction](A17_EFFICIENCY_REPRODUCTION.md).
5. `analysis/public_reanalysis_v1/`: source-bound numerical tables, complete schema audit and cost bindings. `A18_INTERFACE/` exports features and true/anchor outputs; no network was trained.

Please assess the attribution of savings, preservation of quality, fairness of the matched trajectories, source/array bindings, and the distinction between local GN fidelity and final material error. Do not reinterpret this follow-up as a new selector, deterministic certificate, universal acceleration result, or new physical holdout campaign.

## Publication identity

Historical scientific baseline: `f5bd7d0aa0e229110c425f1becb6384e09663196`. Efficiency release tag: `a17-efficient-verification-2026-10-04`. The zip and this directory preserve all 131 admitted package files byte for byte. Publication-only navigation and receipts are additional files, outside the zip manifest. The supplied package supports evidence review and table-based replotting; it is not a self-contained GPU replay bundle.
