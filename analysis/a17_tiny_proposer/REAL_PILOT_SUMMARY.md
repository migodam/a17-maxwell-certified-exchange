# Fixed six-run real priority pilot

COMPLETED once with unchanged frozen trainer/config: two architectures × three seeds ×100 epochs, CPU FP64 one thread. Exactly36 complete receiver-neighborhood round0 dictionary traces;43,108 feasible candidates. Split train2007/2012 (18checkpoints), validation2001 (9), heldout test2014 (9). Validation did not select model/epoch; test only evaluated after each trained model was serialized. All six receipts/models/loss curves retained, no tuning/retry.

| Architecture | Seed | Train2007 | Train2012 | Validation2001 | Test2014 |
|---|---:|---:|---:|---:|---:|
|tiny_mlp_32_16|20261002|0.679206|0.947706|0.911588|0.988459|
|tiny_mlp_32_16|20261003|0.992289|0.995049|0.961906|0.988461|
|tiny_mlp_32_16|20261004|0.975397|0.979843|0.977467|0.991617|
|group_mean_32_16|20261002|0.606952|0.836859|0.909502|0.765571|
|group_mean_32_16|20261003|0.624636|0.975794|0.981503|0.759893|
|group_mean_32_16|20261004|0.939917|0.958052|0.909124|0.980288|
|Qhat top2 baseline|fixed|0.995903|0.995344|0.997023|0.998850|

Capture is sum best(full-dictionary Q,no-op) across an object's nine common receiver checkpoints in the denominator, and sum best(predicted top2 Q,no-op) in the numerator. Finalist labels were already-paid offline labels, not fresh physical acceptance. Candidate counts are not independent generalization samples; a single heldout object does not establish broad safety or universality. No seed or architecture is selected here. Root owns interpretation and gates.

The baseline retains higher heldout capture than every learned run in this fixed pilot. This descriptive outcome does not establish that all learned proposals fail or that a different method would improve; no tuning or additional model was attempted.

Feature provenance: actual runtime receipt generator sources were matched to deployment ZIP SHA. The endpoint batch forms reduced Z from QR(SU) and Galerkin material injection/propagation F, evaluates2||Z||F² excludingLambda, and run_exchange stores evaluated.info_trace. This is reduced-model information_trace, not evaluator full/projected information. Qhat is independently richer-anchor gain; child step norm/stability/trace are reduced endpoint quantities. FullQ labels are targets/evaluation only. Thirty-six canary injections of all forbidden fields left features unchanged; normalization fits only train objects. Source/config SHA still match pre-real frozen manifest.

DATASET_MANIFEST_FINAL.json binds each full trace's SHA, complete Cartesian1swap coverage, original round0/baseU/state/pool identity, terminal result/status, runtime/source and prior-paid setup/checkpoint cost receipts. Full dictionary includes preserved non-feasible records. Paid-feature source binding does not imply acquisition cost was measured separately or was free.

Measured CPU: six training intervals sum 9.412759s; six runs' inference on all36checkpoints sum 0.024527s; shared feature-array construction 0.096542s; shared train normalization fit 0.000758s; manifest binding 1.790952s. These are component timings, not whole-job/all-in elapsed time. Feature array construction excludes prior anchor/child physics acquisition. Full feature acquisition, fresh physical finalist validation and all-in speedup remain null/NOT_MEASURED. No speedup or generalized safety claim.

Outputs: real_runs_final/SIX_RUN_INDEX.json, six .pt/.json receipts, QHAT_TOP2_BASELINE.json, normalization.json, AUTHORIZATION_BINDING.json and OBJECT_TOP2_CAPTURE.csv. Detailed legal/count/cost/hash checks in REAL_PILOT_SUMMARY.json. No new Maxwell/fulladjoint/tangent actions, GPU, remote operations or changes to frozen source/config/data.

Final cached-only postprocessing completed after the pilot:36 checkpoints/3665 persisted-spectrum moves resolved by paired-ranking tool (fixed subset and biased teacher-best augmentation separate). Fixed full-reference direction decomposition passed on6 Gaussian states/72 policy endpoints, maximum archived RGN-gain relative discrepancy6.500e-11;6 voxel states remainPROJECTED_ONLY, no full directional decomposition. Summary in FINAL_CACHED_POSTPROCESS_SUMMARY.json. One broad directory scan also encountered the known historical incomplete directory; INITIAL_ALL_DIRECTORY_SCAN.json retains that outcome, and final12-state selection is bound to this pilot manifest rather than directory recency. No historical record was changed.
