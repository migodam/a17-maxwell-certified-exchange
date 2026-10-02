# Reproduction and evidence inspection

## Public scope

Read raw outcomes, failed/closed attempts, costs and exact source receipts; recreate figures from saved tables; run new independent portable CPU algebra. Forty-two historical vendor modules are external. GPU Maxwell replay additionally needs those exact hash-identified kernels and the saved inputs/environment. A portable test PASS is not a historical Maxwell rerun.

## Evidence map

- A17_R1/execution/gpu_attempts.jsonl: 17 attempts, all paired.
- A17_R1/results/object_*/: 12 complete states, 36 cells, 180 policies. The successful near-contact early state is object_2007_early_portable; object_2007_early is a preserved failed start.
- analysis/a17_pilot_analysis/analysis_output_final/per_policy_main.csv and object_summed_ratios.csv: within-object paired aggregation.
- analysis/a17_pilot_analysis/analysis_output_final/same_checkpoint_score_regret.csv: one-decision ranking/coverage, not multi-action terminal loss.
- A17_R1/results/timing_* and analysis/a17_matched_timing_analysis/: actual physical warm timing, formal/prewarm/evaluation costs separate.
- A17_R1/figures/exchange/: saved plotted scalars and source provenance. reproduce/replot_published.py reads these scalars directly, without historical private kernels.
- information and learned evidence: projection scope, feature permissions, all six runs.

The original new entry is A17_R1/code/run_exchange.py with A17_R1/A17_RUN_CONFIG.json, inputs and SOURCE_MANIFEST.json. The source/runtime association audit distinguishes observed source hashes from later analysis and tests. Do not restore old A16 queues. No physics replay is executed during publication.

## Portable checks and figures

With Python, NumPy and Matplotlib already available, from the public repository root:

```sh
python reproduce/portable_algebra.py
python reproduce/replot_published.py --data A17_R1/figures/exchange/figure_data.csv --out reanalysis_figures
```

The first command checks independently authored real-variable algebra. The second redraws the final data figures. Neither recreates a full Maxwell run. All four document sources are under manuscript/. They require a TeX engine with the declared fonts and packages; the supplied PDFs are the reviewed outputs.

## Integrity

New gate, pilot, accounting and information source epochs are explicit. The extension uses V5; matched timing binds that terminal source and unchanged core hashes. Public eligible Python/NPZ files are byte-copied. Registered metadata locators have exact transformations, original/public hashes and unchanged scientific fingerprints; there is no whole-source regex cleaning.

Private terminal archive SHA256:
extension e02ce8a36295185b578c8511588128c2efbccbd58e2ff741036a97cac34bd0a7
timing delta ebb165add9b313430828a24a3d73828018e63840188475e2710a79beb84571ab

Those private bulk archives are not uploaded. Permitted individual records and manifests preserve lineage. Exclusions are disclosed, not replaced with altered executable expressions.
