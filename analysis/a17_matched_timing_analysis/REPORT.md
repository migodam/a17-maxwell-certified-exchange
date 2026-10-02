# A17 matched physical timing: persisted read-only analysis

Both snapshots are COMPLETE: 45 timed samples plus 9 separately paid prewarm trials per object. Five rotated repetitions exist for every actual k∈{4,8,16} and method; 108 journal NPZ files pass file SHA, FP64/complex128 array SHA, finiteness, rank and basis checks. Sources/imports are unchanged start→end and match the present local files. No physics, timing, GPU, SSH, source mutation or scientific adjudication was performed.

|object|actual k|control|warm selector median [Q1,Q3], s|offline evaluation median [Q1,Q3], s|full objective gain|accepted|selector RHS|
|---|---:|---|---|---|---:|---:|---:|
|2007|4|legacy16 fixed one-pass|0.137579 [0.119405, 0.139619]|0.095502 [0.092574, 0.100086]|0|0–0|0–0|
|2007|4|same12 one decision|0.328122 [0.324828, 0.333298]|0.094914 [0.094079, 0.103372]|4.45047382e-06|1–1|18–18|
|2007|4|conditional12 max3|0.594083 [0.587099, 0.607147]|0.096428 [0.095249, 0.102188]|4.45047382e-06|1–1|30–30|
|2007|8|legacy16 fixed one-pass|0.113492 [0.107764, 0.133952]|0.097845 [0.092774, 0.104438]|0|0–0|0–0|
|2007|8|same12 one decision|0.502559 [0.495073, 0.526009]|0.105055 [0.095128, 0.107021]|1.5241863e-05|1–1|18–18|
|2007|8|conditional12 max3|1.417685 [1.394043, 1.442036]|0.107150 [0.098209, 0.110318]|2.75089234e-05|2–2|42–42|
|2007|16|legacy16 fixed one-pass|0.150775 [0.148101, 0.154854]|0.100644 [0.096832, 0.108158]|-2.16891353e-07|1–1|0–0|
|2007|16|same12 one decision|0.937260 [0.926552, 0.945698]|0.111785 [0.105119, 0.117318]|7.36531674e-06|1–1|18–18|
|2007|16|conditional12 max3|2.690279 [2.673276, 2.728815]|0.110924 [0.108997, 0.113823]|8.83801221e-06|3–3|42–42|
|2012|4|legacy16 fixed one-pass|0.195596 [0.190103, 0.198944]|0.085220 [0.084108, 0.087462]|0.00201897872|1–1|0–0|
|2012|4|same12 one decision|0.332848 [0.331442, 0.337536]|0.083855 [0.080886, 0.084528]|0.00204458386|1–1|18–18|
|2012|4|conditional12 max3|0.891049 [0.890644, 0.900735]|0.087235 [0.083444, 0.087809]|0.00209179807|2–2|42–42|
|2012|8|legacy16 fixed one-pass|0.206910 [0.206216, 0.210556]|0.093120 [0.089238, 0.093156]|0|0–0|0–0|
|2012|8|same12 one decision|0.596274 [0.588742, 0.619008]|0.089113 [0.088376, 0.090137]|4.04416612e-06|1–1|18–18|
|2012|8|conditional12 max3|1.152519 [1.145977, 1.181334]|0.094381 [0.093985, 0.097271]|4.04416612e-06|1–1|30–30|
|2012|16|legacy16 fixed one-pass|0.219012 [0.214352, 0.245060]|0.082888 [0.082594, 0.093973]|0|0–0|0–0|
|2012|16|same12 one decision|1.215058 [1.183295, 1.232091]|0.083679 [0.079882, 0.084281]|5.00817119e-05|1–1|18–18|
|2012|16|conditional12 max3|3.399540 [3.388797, 3.400574]|0.084417 [0.080643, 0.084747]|5.36015209e-05|3–3|42–42|

Legacy uses the fixed at-most16 proposals with the original family order and original-column deletion schedule, accepts sequential positive anchor gains, and does no directed verification. It is a cached receiver one-pass control, **not full old S8**. Same12 uses the first A17 conditional neighborhood and at most one directed accepted decision. Conditional12 regenerates the neighborhood after acceptance, uses ≤12 incoming and ≤2 directed finalists per round, and ≤3 moves. Distinct physical RHS budgets are part of these contracts, not a hidden equal-work comparison.

The negative Gaussian k16 legacy gain (−2.168913534e−7) is retained: one anchor-positive move worsened the measured full linearized objective. This is a finite observation of the ungated surrogate rule, not by itself a rule-implementation bug or a scientific adjudication.

## Compatibility and callback evidence

For all 36 compatible one-decision/final-conditional arrays per object, including prewarm, maximum step relative discrepancy with production directed round0/final arrays is 6.73e-14; basis span residual is at most 3.78e-15. These comparisons allow QR phase changes. Gaussian original production used CPU small-material solves; matched CUDA outputs agree numerically, but production CPU wall is not a CUDA selector-time baseline. Voxel original production and matched endpoint devices are CUDA.

Each directed accepted-Q sum telescopes to the separately evaluated initial-minus-final objective within 2.35e−18 across the persisted trials. Final output arrays are finite; k and rank agree, and the largest orthogonality residual is below 3.75e-15. Final chosen logical IDs/all intermediate endpoint arrays are not saved in these timing journals, so this does not independently replay every accepted identity.

The receipts declare PHYSICAL_CALLBACK/cuda_physical_Maxwell_return_FP64_host, one shared frozen model/state, no dense-fullJ construction, no full-reference step optimization and zero cache-label ranking. Actual native RHS match wrapper charges. Source `portable_timing_driver_v2.py:39–51` installs frozen(...,"cuda") → bundle["ev"].j; the fixture branch is separate at lines56–69 and the receipts explicitly mark test_callback false. `portable_cached_onepass_v2.py:55–73` accepts real FP64 directions and charges P×columns. `operators.py:202–205` expands shared real material coefficients and performs full state JVP, packing every illumination; `run_exchange.py:53–56` uses real residual, ell and lambda. P=6 and r has 1536 packed coordinates for both caches; ell has 54 Gaussian/3456 voxel real coordinates. No new adjoint action is used by this j-only path. Native receipts have no adjoint-labelled action keys/events and all solve RHS are accounted as tangent RHS; absence of sparse keys is not independent hardware instrumentation.

## Fees, history and cold limit

- 2007: fresh shared physical state 5.433814s (one full LU and six forward RHS), cache read 0.049250s, adapter 0.031796s. Paid prewarm selector/evaluation = 7.878886/0.858837s. All54 trials have 156 full tangent action/solve calls and 1656 RHS: selection1008 + offline evaluation648. No additional LU/full L materialization in those trials.
- 2007 historical acquisition: physical 6.049322s; public workspace 11.619840s, with nested anchor 6.217750s, pool 1.477701s and projection 3.915057s. Historical offline reference verification 0.214647s; public information probe 0.000000s/0 RHS. These are old receipt values, not new independent cold measurements, and nested values must not be added twice.
- 2012: fresh shared physical state 5.726768s (one full LU and six forward RHS), cache read 0.136241s, adapter 0.410196s. Paid prewarm selector/evaluation = 9.596325/0.885670s. All54 trials have 156 full tangent action/solve calls and 1656 RHS: selection1008 + offline evaluation648. No additional LU/full L materialization in those trials.
- 2012 historical acquisition: physical 5.601325s; public workspace 12.261021s, with nested anchor 6.540576s, pool 1.576060s and projection 4.082667s. Historical offline reference verification 0.238252s; public information probe 0.567904s/96 RHS. These are old receipt values, not new independent cold measurements, and nested values must not be added twice.

Historical Gaussian receipt lacks an explicit dense-J acquisition time field; this gap stays missing. Historical references/full probes are not selector time. Current offline evaluation (two endpoints,12 RHS per sample) starts after selector wall stops. `method_by_k.csv` retains basis/endpoint/anchor/Jx/Jd phase medians and endpoint transfer-byte estimates separately; nested endpoint/model/callback views overlap and must not be added to total job occupation. Native physical internal transfer bytes are not fully instrumented. All standalone cold_wall_s/cold_speedup fields remain null; fresh shared state time alone is not end-to-end cold acquisition or a cold speedup.

## Failure and stage-timer audit

No job/sample/prewarm failure, isolated endpoint unit/factorization failure, or nonfinite saved array is observed. Full available endpoint statistics and trace failures were scanned; zero triggering all-current-fail/all-Schur-fail units are observed in these complete snapshots. The known early returns before current/Schur stage mark remain in `endpoint_batch.py:134` and `removal_core_batch.py:162`. A future all-failed unit can omit that phase wall even though total_evaluate_wall/counters/failed occupation are charged. This audit does not close the code gap or treat an unidentified/general trigger as zero. No driver rule or source was modified.

Scientific interpretation remains with the root. Figures display warm timing and signed gains only; no offline time is renamed selector time.

Artifacts: `SUMMARY.json`, `method_by_k.csv`, `comparison.csv`, `INPUT_PROVENANCE.json`; figures `warm_wall.png/.pdf`, `objective_gain.png`. Source/receipt/NPZ hashes reside in the provenance and summary files.
