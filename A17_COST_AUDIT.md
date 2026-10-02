# Complete cost audit

**Remote occupation is 4889.515 s = 81.49 min = 1.3582 h, below the 12 h ceiling.** All 17 attempts are paired and closed, including 2 failed starts. Peak sampled memory is 1872 MiB on an 8188 MiB device; there is no OOM or resource stop.

Occupation includes CPU work, teacher and offline work, failures/retries and prewarm; it is not GPU-active duration. Nested component wall times must not be added twice.

## Matched physical timing

Two early states use actual CUDA FP64 Maxwell Jd. Each has 45 formal + 9 paid prewarm samples, with 5 rotated repetitions per k/policy. There are 1656 tangent RHS per object: 1008 for selection + 648 for separate offline evaluation. Compatible step error is ≤6.73e−14. The original Gaussian production small solve uses CPU; the matched run uses CUDA. Do not merge their wall times.

|Space|Policy|k = 4 median s|k = 8|k = 16|
|---|---|---:|---:|---:|
|Gaussian|Cached fixed one-pass|.137579|.113492|.150775|
|Gaussian|One directed decision|.328122|.502560|.937260|
|Gaussian|Conditional ≤3|.594083|1.417685|2.690279|
|Voxel|Cached fixed one-pass|.195596|.206910|.219012|
|Voxel|One directed decision|.332848|.596274|1.215058|
|Voxel|Conditional ≤3|.891049|1.152519|3.399540|

The one-pass control has distinct semantics and is not the complete old S8 algorithm. Conditional verification costs more and buys full-step fidelity; there is no net acceleration.

Fresh common physical setup takes 5.4338 s/5.7268 s. Historical public workspace times of 11.6198 s/12.2610 s include nested anchor, pool and projection costs; these are old receipts, not independently measured cold costs. Independent cold_wall_s/speedup = null. Shared, warm and reconstructed standalone fees remain distinct.

All-failed current/Schur batches can omit a phase timer on early return. This was not triggered in the matched samples; full attempt wall time and counters charge the work, while the general gap stays OPEN. Implicit kernel transfer bytes are incompletely instrumented. CPU training or thin-SVD optimization is not a whole-pipeline speedup.

The old paid paired 1.416×PCG/RHS 66→342 result remains.

Sources: execution/gpu_attempts.jsonl, ledgers/closed_attempt_costs.csv, ledgers/matched_costs.csv, analysis/matched/REPORT.md and raw timing journals.

